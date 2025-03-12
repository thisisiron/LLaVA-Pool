import os
import argparse
import torch
import re
import csv
import shutil
from typing import Dict, Tuple
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor
from llavapool.data import Role


def _parse_vqa(example: Dict[str, str], prompt="Answer:", is_fewshot=False) -> Tuple[str, str]:
    """
    Parse a VQA example and return prompt and answer.
    
    Args:
        example: Dictionary containing question, choices, and answer
        prompt: Prompt template
        is_fewshot: Whether the example is a few-shot example
        
    Returns:
        (prompt, answer) tuple
    """
    # Dynamically identify choices
    choices_pattern = re.compile(r'choice_([a-z])', re.IGNORECASE)
    choices = {match.group(1).upper(): example[key] for key, match in 
               ((k, choices_pattern.match(k)) for k in example) if match}
    
    if not choices:
        choices = {ch: example[ch] for ch in "ABCD" if ch in example}

    prompt = prompt.replace("<image>\n", "").replace("<image>", "").strip()
    
    formatted_prompt = prompt.format(
        question=example["question"],
        A=choices.get("A", ""),
        B=choices.get("B", ""),
        C=choices.get("C", ""),
        D=choices.get("D", ""),
    )
    
    return formatted_prompt, example["answer"]


def convert_to_vqa_format(data, examples, prompt):
    """
    Convert data to VQA format.
    
    Args:
        data: Data to convert
        examples: Few-shot examples
        prompt: Prompt template
        
    Returns:
        Converted data
    """
    messages = []

    # Add few-shot examples
    for example in examples:
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

    # Add actual question
    question, answer = _parse_vqa(data, prompt)
    messages.append({
        "role": Role.USER,
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]
    })
    
    return {
        "_prompt": messages, 
        "_images": data['image'],
        "_question": question,
        "_answer": data['answer']
    }


def map_function(data, prompt, train_dataset, few_shot):
    """
    Dataset mapping function. Converts data with few-shot examples.
    
    Args:
        data: Data to convert
        prompt: Prompt template
        train_dataset: Training dataset (for few-shot example selection) 
        few_shot: Number of few-shot examples
        
    Returns:
        Converted data
    """
    if train_dataset is not None and few_shot > 0:
        import random
        random.seed(42)
        indices = random.sample(range(len(train_dataset)), few_shot)
        examples = train_dataset.select(indices)
    else:
        examples = []
    return convert_to_vqa_format(data, examples, prompt)


def collate_fn(batch, processor):
    """
    Convert batch data to model input format.
    
    Args:
        batch: Batch data
        processor: Text/image processor
        
    Returns:
        Batch converted to model input format
    """
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
    """
    Move batch to specified device.
    
    Args:
        batch: Batch data 
        device: Device to move to
        
    Returns:
        Batch moved to device
    """
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="K-DTCBench Evaluation Script")
    
    # Model related arguments
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        required=True,
        help="Model path"
    )
    parser.add_argument(
        "--use_flash_attn", 
        action="store_true", 
        help="Whether to use flash attention"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16", 
        choices=["bf16", "fp16", "fp32"],
        help="Precision for model inference"
    )

    # Dataset related arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NCSOFT/K-DTCBench",
        help="Dataset name to use"
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )

    # Evaluation related arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--save_incorrect_images",
        action="store_true",
        help="Whether to save images of incorrect predictions"
    )

    return parser.parse_args()


def main():
    """
    Main function to run K-DTCBench dataset evaluation
    """
    # Parse arguments
    args = parse_args()
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
    )
    tokenizer = processor.tokenizer
    
    # Adjust image processor max pixels if needed
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = 28 * 28 * 1280
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else 
                   (torch.float16 if args.precision == "fp16" else torch.float32),
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device)

    model = model.eval()

    # Set prompt
    base_prompt = (
        "<image>\n"
        "Question:\n{question}\n"
        "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
        "Answer with just the option letter from the given choices.\n"
    )

    # Load dataset
    print(f'Loading dataset: {args.dataset_name}')
    dataset = load_dataset(args.dataset_name)
    
    # Convert dataset
    column_names = dataset["test"].column_names.copy()
    processed_dataset = dataset.map(
        partial(
            map_function, 
            prompt=base_prompt, 
            train_dataset=dataset.get("train", None), 
            few_shot=args.few_shot
        ),
        desc="Converting dataset to VQA format",
        remove_columns=column_names,
        batched=False
    )

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset=processed_dataset["test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    # Create directory for saving images
    images_dir = os.path.join(args.save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Perform evaluation
    merged_outputs = defaultdict(list)
    batch_idx = 0
    image_paths = []  # List of saved image paths
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Running model inference"):
            answer = inputs.pop('answer')
            question = inputs.pop('question')
            inputs = to_device(inputs, args.device)
            
            # Convert images to model's dtype
            if args.precision == "bf16":
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            elif args.precision == "fp16":
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            
            # Model inference
            pred = model.generate(
                **inputs, 
                max_new_tokens=128,  # Reduced number of generated tokens
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode output tokens
            outputs = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], pred)]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Save current batch images
            for i in range(len(question)):
                img_index = batch_idx * args.batch_size + i
                img_path = f"k-dtc_{img_index}.jpg"
                full_img_path = os.path.join(images_dir, img_path)
                
                # Process current image (from original data)
                original_img = processed_dataset["test"][img_index]["_images"]
                if isinstance(original_img, str):
                    # Copy if it's an image path
                    if os.path.exists(original_img):
                        shutil.copy2(original_img, full_img_path)
                else:
                    # Save if it's a PIL.Image object
                    if isinstance(original_img, Image.Image):
                        original_img.save(full_img_path)
                
                relative_path = os.path.join("images", img_path)
                image_paths.append(os.path.join(args.save_dir, relative_path))
            
            # Save results
            merged_outputs['question'] += question
            merged_outputs['answer'] += answer
            merged_outputs['pred'] += outputs
            
            batch_idx += 1
            torch.cuda.empty_cache()
    
    # Calculate accuracy
    cnt = 0
    is_correct = []  # Track correct answers
    
    for i in range(len(merged_outputs['question'])):
        target = merged_outputs['answer'][i].strip().lower()
        pred = merged_outputs['pred'][i].strip().lower()
        
        # Correct if exact match or first letter matches and is a valid option
        correct = False
        if pred == target:
            correct = True
            cnt += 1
        elif len(pred) >= 1 and pred[0] in 'abcd' and pred[0] == target:
            correct = True
            cnt += 1
        else:
            print(target, pred)
        
        is_correct.append(correct)
    
    accuracy = cnt / len(merged_outputs['question'])
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save results to CSV file
    os.makedirs(args.save_dir, exist_ok=True)
    csv_file_path = os.path.join(args.save_dir, f"kdtcbench_evaluation_results.csv")
    
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        writer.writerow(["Question", "Answer", "Prediction", "Image Path", "Correct"])
        
        # Write data
        for i in range(len(merged_outputs['question'])):
            writer.writerow([
                merged_outputs['question'][i],
                merged_outputs['answer'][i],
                merged_outputs['pred'][i],
                image_paths[i],
                "O" if is_correct[i] else "X"
            ])
    
    print(f"Results saved to {csv_file_path}")
    print(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()