from dataclasses import dataclass
from collections import defaultdict
import os
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

from datasets import DatasetDict, load_dataset
from datasets import Dataset, IterableDataset
from transformers import logging
from transformers import (
    PreTrainedTokenizer,
    TrainingArguments,
    ProcessorMixin,
)

from .dataset_config import (
    get_dataset_config,
)
from .collator import SFTDataCollatorWith4DAttentionMask
from ..utils.constants import IGNORE_INDEX


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
    dataset_config: "DatasetConfig",
    data_args: "DataArguments",
):
    """
    Converts a ShareGPT-format conversation example into a standardized internal format used for training.

    This function:
    1. Takes a conversation example in ShareGPT format and transforms it into a consistent internal representation
    2. Handles various message roles (user, assistant, system, function, observation)
    3. Validates the turn-taking pattern of the conversation (user/observation followed by assistant/function)
    4. Processes any included media (images and videos) by joining them with the dataset directory path
    5. Separates the conversation into prompt (all messages except last) and response (last message)

    Parameters:
    - example: dict - The input conversation example in ShareGPT format
    - dataset_config: DatasetConfig - Configuration for processing the dataset
    - data_args: DataArguments - Arguments for data processing

    Returns:
    A dictionary containing:
    - _prompt: List of messages forming the conversation context
    - _response: List containing the final response message
    - _images: List of image paths or None
    - _videos: List of video paths or None
    - _system: System message content or empty string
    - _tools: Tools content or empty string if not specified in config
    """
    
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
    messages = example[dataset_config.common.messages]

    if dataset_config.sharegpt.system_tag and messages[0][dataset_config.sharegpt.role_tag] == dataset_config.sharegpt.system_tag:
        system = messages[0][dataset_config.sharegpt.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_config.common.system] if dataset_config.common.system else ""

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
    
    images = None
    if dataset_config.common.images is not None:
        if isinstance(example[dataset_config.common.images], str):
            images = [os.path.join(data_args.dataset_dir, example[dataset_config.common.images])]
        elif isinstance(example[dataset_config.common.images], list):
            if len(example[dataset_config.common.images]) == 0:
                logger.warning(f"Empty image list in example: {messages}")
                images = None
            else:
                images = []
                for idx in range(len(example[dataset_config.common.images])):
                    images.append(os.path.join(data_args.dataset_dir, example[dataset_config.common.images][idx]))

    videos = None
    if dataset_config.common.videos is not None:
        if isinstance(example[dataset_config.common.videos], str):
            videos = [os.path.join(data_args.dataset_dir, example[dataset_config.common.videos])]
        elif isinstance(example[dataset_config.common.videos], list):
            videos = []
            for idx in range(len(example[dataset_config.common.videos])):
                videos.append(os.path.join(data_args.dataset_dir, example[dataset_config.common.videos][idx]))

    return {
        "_prompt": prompt,
        "_response": response,
        "_images": images,
        "_videos": videos,
        "_system": system,
        "_tools": example[dataset_config.common.tools] if dataset_config.common.tools else "",
    }


def convert_dataset(
    dataset,
    dataset_config: "DatasetConfig",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    format: str = "sharegpt",
):
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        )
    column_names = list(next(iter(dataset)).keys())
    
    if format == "sharegpt":
        formatting_func = convert_sharegpt

    dataset = dataset.map(
        lambda x: formatting_func(x, dataset_config, data_args),
        batched=False,
        desc="Converting dataset to ShareGPT format",
        remove_columns=column_names,
        **kwargs
    )
    return dataset
    

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


def _encode_supervised_example(
    prompt: List[Dict[str, str]],  # [{"role": "user", "content": "text"}, ...]
    response: List[Dict[str, str]],  # [{"role": "assistant", "content": "text"}, ...]
    images: Optional[str] = None,
    videos: Optional[str] = None,
    system: Optional[str] = None,
    tools: Optional[str] = None,
    converter: str = None,
    tokenizer: "PreTrainedTokenizer" = None,
    processor: "ProcessorMixin" = None,
    cutoff_len: int = 1024,
    train_on_prompt: bool = False,
    mask_history: bool = False,
    data_args: "DataArguments" = None,
):
    input_ids, labels = [], []
    encoded_pairs = converter.encode_multi_turn(
        prompt + response, 
        images, 
        videos,
        system=system,
        tools=tools,
    )
    total_length = 1 if converter.conversation.efficient_eos else 0

    if mask_history:
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
        elif converter.conversation.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids
        
        if mask_history:
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label
    
    if converter.conversation.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]
    
    return input_ids, labels


def preprocess_superivsed_dataset(
    examples: Dict[str, List[Any]],
    converter,
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            converter=converter,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
            data_args=data_args,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs
    

def get_superivsed_dataset(
    dataset,
    converter,
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
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
        lambda x: preprocess_superivsed_dataset(x, converter, tokenizer, processor, data_args),
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


from datasets import concatenate_datasets
def load_dataset_module(
    converter,
    data_args: "DataArguments",
    model_args: "ModelArguments",
    training_args: "TrainingArguments",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin" = None,
    stage: str = "sft",
):
    
    datasets = []
    for dataset_name in data_args.dataset:
        dataset_config = get_dataset_config(data_args.dataset_dir, dataset_name)

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
            cache_dir=model_args.cache_dir,
            # token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            trust_remote_code=True,
        )
        
        with training_args.main_process_first(desc="load dataset"):
            dataset = convert_dataset(dataset, dataset_config, data_args, training_args, format=dataset_config.formatting)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = concatenate_datasets(datasets)

    # dataset_config = get_dataset_config(data_args.dataset_dir, data_args.dataset)

    # logger.info(f"Loading dataset {data_args.dataset}...")
    # data_path, data_name, data_dir = None, None, None
    
    # data_files = []
    # data_path = os.path.join(data_args.dataset_dir, dataset_config.file_name)
    # if os.path.isfile(data_path):  # is file
    #     data_files.append(data_path)
    #     file_format = data_path.split(".")[-1]
    #     data_path = file_format
    # else:
    #     raise ValueError(f"File {data_path} not found.")
    
    # dataset = load_dataset(
    #     path=data_path,
    #     name=data_name,
    #     data_dir=data_dir,
    #     data_files=data_files,
    #     split=dataset_config.split,
    #     cache_dir=model_args.cache_dir,
    #     # token=model_args.hf_hub_token,
    #     streaming=data_args.streaming,
    #     # trust_remote_code=True,
    # )
    # with training_args.main_process_first(desc="loading dataset..."):
    #     dataset = convert_dataset(dataset, dataset_config, data_args, training_args, format=dataset_config.formatting)

    
    with training_args.main_process_first(desc="pre-processing dataset..."):
        dataset = get_superivsed_dataset(dataset, converter, tokenizer, processor, data_args, training_args)

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

        for data in dataset_dict['train']:
            assert len(data['input_ids']) == len(data['labels']) == len(data['attention_mask'])

        return dataset_module