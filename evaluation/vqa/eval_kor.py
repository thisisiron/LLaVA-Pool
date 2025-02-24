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


def _parse_vqa(example: Dict[str, str], prompt="Answer:", is_fewshot=False) -> Tuple[str, str]:
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

    prompt = prompt.replace("<image>\n", "").replace("<image>", "").strip()
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
        question, answer = _parse_vqa(example, prompt, is_fewshot=True)
        messages.append({
            "role": Role.USER,
            "content": [
                {"type": "text", "text": question},
            ]
        })
        messages.append({
            "role": Role.ASSISTANT,
            "content": [
                {"type": "text", "text": answer},
            ]
        })

    # if "<image>" not in data['question']:
    #     data['question'] = f"<image>/n {data['question']}"

    question, answer = _parse_vqa(data, prompt)
    messages.append({
        "role": Role.USER,
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]
    })
    
    # messages.append({
    #     "role": Role.ASSISTANT,
    #     "content": [
    #         {"type": "text", "text": answer},
    #     ]
    # })

    return {
        "_prompt": messages, 
        "_images": data['image'],
        "_question": question,
        "_answer": data['answer']
    }

def collate_fn(batch, processor):
    prompts = []
    images = []
    questions = []
    answers = []
    for data in batch:
        prompts.append(data["_prompt"])
        images.append(data["_images"])
        questions.append(data["_question"])
        answers.append(data["_answer"])

    text = processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt"
    )
    inputs.update({
        "answer": answers,
        "question": questions,
    })

    return inputs


def to_device(batch, device='cuda'):
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def main():
    
    model_args, data_args, eval_args, finetung_args = get_eval_args()

    tokenizer_module = load_tokenizer_and_processor(model_args)  # processor exists or not
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    
    model = load_model(tokenizer, model_args, finetung_args)
    # model = model.eval()

    # from transformers import AutoModel, AutoTokenizer
    # path = "OpenGVLab/InternVL2_5-4B"
    # path = "output/internvl2_5-8b/full/test/checkpoint-10000"
    # model = AutoModel.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     use_flash_attn=True,
    # trust_remote_code=True).eval().cuda()
    
    # os.makedirs(eval_args.save_dir, exist_ok=True)

    base_prompt = (
        "<image>\n"
        "Question:\n{question}\n"
        "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
        "주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.\n"
    )

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

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset["test"],
        # sampler=InferenceSampler(len(dataset)),
        batch_size=eval_args.batch_size,
        num_workers=0,  # eval_args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    merged_outputs = defaultdict(list)
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            answer = inputs.pop('answer')
            question = inputs.pop('question')
            inputs = to_device(inputs, 'cuda')
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            print(inputs['pixel_values'].shape)

            pred = model.generate(
                **inputs, 
                max_new_tokens=4092,
                eos_token_id=tokenizer.eos_token_id,
            )

            outputs = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], pred)]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            merged_outputs['question'] += question
            merged_outputs['answer'] += answer
            merged_outputs['pred'] += outputs
            
    cnt = 0
    for i in range(len(merged_outputs['question'])):
        target = merged_outputs['answer'][i].strip().lower()
        pred = merged_outputs['pred'][i].strip().lower()
        if pred == target:
            cnt += 1
        elif len(pred) >= 2 and pred[0] in 'abcd':
            if pred[0] == target:
                cnt += 1

    print(f"Acc@1: {cnt / len(merged_outputs['question'])}")

    import csv

    # Define CSV file path
    csv_file_path = "evaluation_results.csv"

    # Save results to CSV
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["question", "answer", "pred"])

        # Write data
        for i in range(len(merged_outputs['question'])):
            # 두 단계의 인덱싱으로 중첩 리스트 내의 첫 번째 질문 텍스트를 추출합니다.
            question_text = merged_outputs['question'][i]
            writer.writerow([question_text, merged_outputs['answer'][i], merged_outputs['pred'][i]])

    print(f"Results saved to {csv_file_path}")


if __name__ == "__main__":
    main()