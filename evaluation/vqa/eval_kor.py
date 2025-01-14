
from llavapool.hparams import get_eval_args
from llavapool.model import load_tokenizer_and_processor, load_model
from llavapool.data import load_converter
from datasets import load_dataset
from datasets import load_dataset
import argparse
import torch
from typing import Dict, Tuple
from llavapool.data import Role
from functools import partial
from collections import defaultdict

from tqdm import tqdm

import re



import random
random.seed(42)

CHOICES = ["A", "B", "C", "D"]

def map_function(data, prompt, train_dataset, few_shot):
    if train_dataset is not None:
        indices = random.sample(range(len(train_dataset)), few_shot)
        examples = train_dataset.select(indices)
    else:
        examples = []
    return convert_to_vqa_format(data, examples, prompt)


def _parse_vqa(example: Dict[str, str], prompt="Answer:") -> Tuple[str, str]:
    r"""
    input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
    output: a tuple of (prompt, response)
    """
    # Identify choices dynamically
    choices_pattern = re.compile(r'choice_([a-z])', re.IGNORECASE)
    choices = {match.group(1).upper(): example[key] for key, match in 
               ((k, choices_pattern.match(k)) for k in example) if match}
    
    if not choices:
        choices = {ch: example[ch] for ch in "ABCD" if ch in example}

    prompt = prompt.format(
        question=example["question"],
        A=choices["A"],
        B=choices["B"],
        C=choices["C"],
        D=choices["D"],
    )
    
    return prompt, example["answer"]


def convert_to_vqa_format(data, examples, prompt):
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

    # if "<image>" not in data['question']:
    #     data['question'] = f"<image>/n {data['question']}"

    question, answer = _parse_vqa(data, prompt)
    messages.append({
        "role": Role.USER,
        "content": question,
    })
    messages.append({
        "role": Role.ASSISTANT,
        "content": answer,
    })

    return {
        "_prompt": messages, 
        "_images": data['image'],
        "_answer": data['answer']
    }

def collate_fn(batch, converter):
    model_inputs = defaultdict(list)
    
    for data in batch:
        prompt = data["_prompt"]
        images = data["_images"]
        answer = data["_answer"]

        input_ids, _ = converter.encode_single_turn(prompt, [images], [])

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["answer"].append(answer)
        model_inputs["images"].append(data["_images"])
        model_inputs["prompt"].append(prompt)

    model_inputs.update(converter.get_visual_inputs(model_inputs["images"], [], input_ids))
    model_inputs.pop("images")
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
    
    return model_inputs


def to_device(batch, device='cuda'):
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def main():
    
    model_args, data_args, eval_args, finetung_args = get_eval_args()

    tokenizer_module = load_tokenizer_and_processor(model_args)  # processor exists or not
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    model = load_model(tokenizer, model_args, finetung_args)

    # os.makedirs(eval_args.save_dir, exist_ok=True)

    base_prompt = (
        "<image>\n"
        "Question:\n{question}\n"
        "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
        "주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.\n"
    )

    # base_prompt = (
    #     "<image>\n"
    #     "{question}\n"
    #     "Options: A: {A} B: {B} C: {C} D: {D}\n"
    #     "주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.\n"
    # )

    print('datasets:', data_args.eval_dataset)
    dataset = load_dataset("NCSOFT/K-DTCBench")

    # column_names = list(next(iter(dataset['test'])).keys())
    column_names = ['image']
    
    dataset = dataset.map(
        partial(
            map_function, 
            prompt=base_prompt, 
            train_dataset=dataset.get("train", None), 
            few_shot=eval_args.few_shot
        ),
        desc="Converting dataset to VQA format",
        remove_columns=column_names,
        batched=False
    )

    converter = load_converter(processor, data_args)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset["test"],
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
            prompt = inputs.pop('prompt')
            inputs = to_device(inputs, 'cuda')

            pred = model.generate(
                **inputs, 
                max_new_tokens=1024,
            )

            outputs = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], pred)]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            merged_outputs.append({
                'pormpt': prompt,
                'pred': outputs[0],
                'answer': answer[0],
            })
    
    cnt = 0
    for item in merged_outputs:
        target = item['answer'].strip().lower()
        pred = item['pred'].strip().lower()
        if pred == target:
            cnt += 1
        elif len(pred) >= 2 and pred[0] in 'abcd':
            if pred[0] == target:
                cnt += 1
    print(f"Acc@1: {cnt / len(merged_outputs)}")
    import pdb;pdb.set_trace()



if __name__ == "__main__":
    main()