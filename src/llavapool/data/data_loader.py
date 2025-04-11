import os
from dataclasses import dataclass
from typing import Literal, Union

from datasets import Dataset, DatasetDict, IterableDataset, concatenate_datasets, load_dataset
from transformers import (
    PreTrainedTokenizer,
    ProcessorMixin,
    TrainingArguments,
    logging,
)

from .dataset_config import (
    get_dataset_config,
)
from .strategy import PairwiseStrategy, SupervisedStrategy


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

    # Processing preference data like dpo
    if (
        dataset_config.preference 
        and isinstance(example[dataset_config.common.chosen], dict) 
        and isinstance(example[dataset_config.common.rejected], dict)
    ):
        chosen = example[dataset_config.common.chosen]
        rejected = example[dataset_config.common.rejected]
        prompt = valid_messages
        response = [
           {
                    "role": tag_to_role[chosen[dataset_config.sharegpt.role_tag]],
                    "content": chosen[dataset_config.sharegpt.content_tag],
                },
                {
                    "role": tag_to_role[rejected[dataset_config.sharegpt.role_tag]],
                    "content": rejected[dataset_config.sharegpt.content_tag],
                }
        ]
    else:
        # Standard case
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


def process_dataset_with_strategy(
    dataset,
    converter,
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    stage: str = "sft",
):
    """
    Process dataset with appropriate strategy based on training stage.
    
    Parameters:
    - dataset: The input dataset
    - converter: Template converter for formatting inputs
    - tokenizer: Tokenizer for processing text
    - processor: Processor for handling multimodal inputs
    - data_args: Arguments for data processing
    - training_args: Arguments for training
    - stage: Training stage (sft, rm, dpo)
    
    Returns:
    The processed dataset
    """
    if dataset is None:
        return dataset
    
    # Select appropriate strategy based on stage
    if stage == "sft":
        strategy = SupervisedStrategy()
    elif stage == "rm":
        strategy = PairwiseStrategy()
    elif stage == "dpo":
        strategy = PairwiseStrategy()
    else:
        logger.warning(f"Unknown stage: {stage}, using SupervisedStrategy as default")
        strategy = SupervisedStrategy()
    
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        lambda x: strategy.preprocess(x, converter, tokenizer, processor, data_args),
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    return dataset


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    """
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    
    Parameters:
    - dataset: The input dataset
    - data_args: Arguments for data processing
    - seed: Random seed for splitting
    
    Returns:
    A DatasetDict containing train and validation sets
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
    
    with training_args.main_process_first(desc="pre-processing dataset..."):
        dataset = process_dataset_with_strategy(dataset, converter, tokenizer, processor, data_args, training_args, stage)

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