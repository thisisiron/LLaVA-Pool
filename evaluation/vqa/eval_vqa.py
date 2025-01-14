import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
# from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/val.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


CHOICES = ["A", "B", "C", "D"]
from typing import Dict, Tuple
from llavapool.data import Role


def _parse_vqa(example: Dict[str, str], prompt="Answer:") -> Tuple[str, str]:
    r"""
    input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
    output: a tuple of (prompt, response)
    """
    candidates = [f"{ch} {example[ch]}" for ch in CHOICES if ch in example]
    return "".join([example["question"]] + candidates + [f"\n{prompt}"]), example["answer"]


def convert_to_vqa_format(data, examples, subject_name, prompt):
    messages = []
    for k in range(len(examples)):
        example = examples[k]
        question, answer = _parse_vqa(example, prompt)
        messages.append({
            "role": Role.USER,
            "content": question,
        })
        messages.append({
            "role": Role.ASSISTANT,
            "content": answer,
        })
    if "<image>" not in data['question']:
        data['question'] = f"<image>/n {data['question']}"
    question, answer = _parse_vqa(data, prompt)
    messages.append({
        "role": Role.USER,
        "content": question,
    })
    messages.append({
        "role": Role.ASSISTANT,
        "content": answer,
    })
    return messages


def collate_fn(batch, converter):

    input_ids, attention_mask, batch_images, answers  = [], [], [], []
    for messages, images, answer in batch:
        prompt_ids, _ = converter.encode_single_turn(messages, images, [])
        input_ids.append(prompt_ids)
        attention_mask.append([1] * len(prompt_ids))
        batch_images.extend(images)
        answers.append(answer)

    visual_inputs = converter.get_visual_inputs(batch_images, [], prompt_ids)
    return {
        'input_ids': torch.tensor(input_ids),  # need to pad to the same length
        'attention_mask': torch.tensor(attention_mask),
        **visual_inputs,
        'answer': answers
    }


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, converter, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.few_shot = few_shot
        self.max_num = max_num
        if few_shot > 0:
            self.train = open(train).readlines()
        self.converter = converter

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        # image, question, question_id, annotation = data['image'], data['question'], data['question_id'], data.get('answer', None)
        
        few_shot_examples = []
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_examples.append(sample)
        
        messages = convert_to_vqa_format(data, few_shot_examples, 'VQA', self.prompt)
        images = [data['image']]
        answer = data.get('answer', None)
        return messages, images, answer

        # prompt_ids, _ = self.converter.encode_single_turn(messages, images, [])
        # visual_inputs = self.converter.get_visual_inputs(images, [], prompt_ids)
        # import pdb; pdb.set_trace()
        # return {
        #     'input_ids': torch.tensor(prompt_ids),
        #     'attention_mask': torch.tensor([1] * len(prompt_ids)),
        #     **visual_inputs
        # }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response


def evaluate_chat_model(data_args, eval_args, args):
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    summaries = []

    for ds_name in data_args.eval_dataset:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=eval_args.few_shot,
            # dynamic_image_size=args.dynamic,
            # use_thumbnail=use_thumbnail,
            # max_num=args.max_num
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            # collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, question_ids, annotations) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
                verbose=True
            )
            answers = [pred]

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question': question,
                        'question_id': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa_val', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava', 'infographicsvqa_test',]:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'question': question,
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id.replace('data/vizwiz/test/', ''),
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                evaluator = TextVQAAccuracyEvaluator()
                annotation = json.load(open(ds_collections[ds_name]['annotation'], 'r'))['annotations']
                question_id2answers = {}
                for item in annotation:
                    question_id = item['question_id']
                    answers = [answer['answer'] for answer in item['answers']]
                    question_id2answers[question_id] = answers
                for item in merged_outputs:
                    item['pred_answer'] = item['answer']
                    item['gt_answers'] = question_id2answers[item['question_id']]
                accuracy = evaluator.eval_pred_list(merged_outputs)
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('python eval/vqa/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python eval/vqa/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
                summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}])
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:
                    dst_file = './data/gqa/testdev_balanced_predictions.json'
                    print('python eval/vqa/convert_gqa_for_eval.py --src ' +
                          results_file + ' --dst ' + dst_file)
                    python_path = 'python'
                    os.system(python_path + ' eval/vqa/convert_gqa_for_eval.py --src ' +
                              results_file + ' --dst ' + dst_file)
                    command = f'cd ./data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../'
                    print(command)
                    accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
                else:
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


# def load_model_and_tokenizer(args):
#     config = AutoConfig.from_pretrained(args.checkpoint)
#     num_hidden_layers = config.num_hidden_layers
#     # device_map = split_model(num_hidden_layers)

#     if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
#         load_class = AutoModelForVision2Seq
#     else:
#         load_class = AutoModelForCausalLM

#     model = load_class.from_pretrained(args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
#         load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit).eval()
#     # kwargs = {'device_map': device_map} if args.auto else {}

#     tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

#     try:
#         processor = AutoProcessor.from_pretrained(args.checkpoint)
#         # patch_processor(processor, config, tokenizer, args)
#     except Exception as e:
#         print(f'Error loading processor: {e}')
#         processor = None

#     if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
#         model = model.cuda()
#     return model, tokenizer, processor

def to_device(batch, device='cuda'):
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

from llavapool.hparams import get_eval_args
from llavapool.model import load_tokenizer_and_processor, load_model
from llavapool.data import load_converter
from datasets import load_dataset
def main(args=None):
    model_args, data_args, eval_args, finetung_args = get_eval_args(args)

    tokenizer_module = load_tokenizer_and_processor(model_args)  # processor exists or not
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    model = load_model(tokenizer, model_args, finetung_args)

    # os.makedirs(eval_args.save_dir, exist_ok=True)

    print('datasets:', data_args.eval_dataset)

    converter = load_converter(processor, data_args)

    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    summaries = []

    # dataset = load_dataset("HuggingFaceM4/ChartQA")

    for ds_name in data_args.eval_dataset:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            # train=dataset['train'],
            # test=dataset['test'],
            prompt=input_prompt,
            few_shot=eval_args.few_shot,
            converter=converter,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            # sampler=InferenceSampler(len(dataset)),
            batch_size=eval_args.batch_size,
            num_workers=0,  # eval_args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, converter=converter),
        )

        merged_outputs = []
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                answer = inputs.pop('answer')
                inputs = to_device(inputs, 'cuda')
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                pred = model.generate(**inputs, max_new_tokens=100)

                outputs = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], pred)]
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                merged_outputs.append({
                    'answer': outputs[0],
                    'annotation': answer,
                })

        import pdb; pdb.set_trace()
        if ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
            relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
            print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
            summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}])
    # assert args.batch_size == 1, 'Only batch size 1 is supported'

    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )
    # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')

    evaluate_chat_model(data_args, eval_args, args)


if __name__ == '__main__':
    main()