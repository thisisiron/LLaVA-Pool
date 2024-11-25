from dataclasses import dataclass
import os
from typing import Dict, List, Literal, Optional, Tuple, Union


from datasets import DatasetDict, load_dataset
from datasets import Dataset, IterableDataset
from transformers import logging
from transformers import (
    PreTrainedTokenizer,
    TrainingArguments,
    ProcessorMixin,
)

from llavapool.data.dataset_config import (
    DatasetConfig,
    get_dataset_config,
)
from llavapool.config.params import (
    DataArguments,
    ModelArguments,
)


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class Role:
    USER: Literal["user"] = "user"
    ASSISTANT: Literal["assistant"] = "assistant"
    SYSTEM: Literal["system"] = "system"
    FUNCTION: Literal["function"] = "function"
    OBSERVATION: Literal["observation"] = "observation"
    
    @classmethod
    def is_valid(cls, role: str) -> bool:
        return role in {field: getattr(cls, field) for field in cls.__annotations__}
 

def convert_sharegpt(
    example: dict,
    dataset_config: DatasetConfig,
    data_args: DataArguments,
):
    import pdb; pdb.set_trace()
    tag_to_role = {
        dataset_config.sharegpt.user_tag: Role.USER,
        dataset_config.sharegpt.assistant_tag: Role.ASSISTANT,
        dataset_config.sharegpt.system_tag: Role.SYSTEM,
        dataset_config.sharegpt.function_tag: Role.FUNCTION,
        dataset_config.sharegpt.observation_tag: Role.OBSERVATION,
    }

    user_tags = (dataset_config.sharegpt.user_tag, dataset_config.sharegpt.observation_tag)
    assitant_tags = (dataset_config.sharegpt.assistant_tag, dataset_config.sharegpt.function_tag)
    conversation_tags = (user_tags, assitant_tags)
    messages = example[dataset_config.sharegpt.messages]

    if dataset_config.sharegpt.system_tag and messages[0][dataset_config.sharegpt.role_tag] == dataset_config.sharegpt.system_tag:
        system = messages[0][dataset_config.sharegpt.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_config.sharegpt.system] if dataset_config.sharegpt.system else ""

    valid_messages = []
    for turn_idx, message in enumerate(messages):
        if message[dataset_config.sharegpt.role_tag] not in conversation_tags[turn_idx % 2]:
            raise ValueError(f"Invalid role tag {message[dataset_config.sharegpt.role_tag]}")
        
        valid_messages.append(
            {
                "role": tag_to_role[message[dataset_config.sharegpt.role_tag]], 
                "content": message[dataset_config.sharegpt.content_tag]
            }
        )
    
    prompt = valid_messages[:-1]
    response = valid_messages[-1:]

    if isinstance(example[dataset_config.images], str):
        images = os.path.join(data_args.dataset_dir, example[dataset_config.images])
    elif isinstance(example[dataset_config.images], list):
        for idx in range(len(example[dataset_config.images])):
            example[dataset_config.images][idx] = os.path.join(data_args.dataset_dir, example[dataset_config.images][idx])
    
    if isinstance(example[dataset_config.videos], str):
        videos = os.path.join(data_args.dataset_dir, example[dataset_config.videos])
    elif isinstance(example[dataset_config.videos], list):
        for idx in range(len(example[dataset_config.videos])):
            example[dataset_config.videos][idx] = os.path.join(data_args.dataset_dir, example[dataset_config.videos][idx])
    
    return {
        "_prompt": prompt,
        "_response": response,
        "_images": images,
        "_videos": videos,
        "_system": system,
        "_tools": example[dataset_config.common.tools],
    }


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


IGNORE_INDEX = -100
def _encode_example(
    prompt: List[Dict[str, str]],  # [{"role": "user", "content": "text"}, ...]
    response: List[Dict[str, str]],  # [{"role": "assistant", "content": "text"}, ...]
    images: Optional[str] = None,
    videos: Optional[str] = None,
    system: Optional[str] = None,
    tools: Optional[str] = None,
    template: str = None,
    tokenizer: PreTrainedTokenizer = None,
    processor: ProcessorMixin = None,
    cutoff_len: int = 1024,
    train_on_prompt: bool = False,
    mask_hisotry: bool = False,
):
    messages = template.mm_converter.process_media_tokens(prompt + response, images, videos, processor)
    
    encoded_pairs = template.encode_multi_turn(
        tokenizer=tokenizer,
        messages=messages,
        system=system,
        tools=tools,
    )
    total_length = 1 if template.efficent_eos else 0

    if mask_hisotry:
        encoded_pairs = encoded_pairs[::-1]

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(
            source_len=len(source_ids),
            target_len=len(target_ids),
            cutoff_len=cutoff_len - total_length,
        )
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficent_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_hisotry and turn_idx != 0:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids
        
        if mask_hisotry:
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label
    
    if template.efficent_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]
    
    return input_ids, labels

    

def preprocess_superivsed_dataset(
    dataset,
    template: str,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    if dataset is None:
        return dataset
    
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )
    
    dataset = dataset.map(
        _encode_example,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    return dataset


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})



def load_dataset_module(
    template: str,
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin = None,
):
    dataset_config = get_dataset_config(data_args.dataset_dir, data_args.dataset)

    logger.info(f"Loading dataset {data_args.dataset}...")
    data_path, data_name, data_dir = None, None, None
    
    data_files = []
    data_path = os.path.join(data_args.dataset_dir, dataset_config.file_name)
    if os.path.isfile(data_path):  # is file
        data_files.append(data_path)
        file_format = data_path.split(".")[-1]
        data_path = file_format
    else:
        raise ValueError(f"File {data_path} not found.")
    
    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_config.split,
        # cache_dir=model_args.cache_dir,
        # token=model_args.hf_hub_token,
        streaming=data_args.streaming,
        # trust_remote_code=True,
    )
    with training_args.main_process_first(desc="load dataset"):
        dataset = convert_sharegpt(dataset, dataset_config, data_args)
    
    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = preprocess_superivsed_dataset(dataset, template, tokenizer, processor, data_args, training_args)

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            # if eval_dataset is not None:
            #     if data_args.streaming:
            #         eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

            #     dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))
            import sys
            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module