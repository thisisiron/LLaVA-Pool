import os
import json
from typing import Optional, Literal

from dataclasses import dataclass, field


@dataclass
class CommonColumns:
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    messages: str = "conversations"


@dataclass
class ShareGPTTags:
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    observation_tag: str = "observation"
    function_tag: str = "function_call"
    system_tag: str = "system"


@dataclass
class DatasetConfig:
    # Basic configs
    dataset_name: str
    file_name: str
    formatting: str = "sharegpt"
    
    # Extra configs
    subset: Optional[str] = None
    split: str = field(default="train")
    num_samples: Optional[int] = None
    
    # Nested dataclasses
    common: CommonColumns = field(default_factory=CommonColumns)
    sharegpt: Optional[ShareGPTTags] = field(default=ShareGPTTags)


def get_dataset_config(
    dataset_dir: str,
    dataset_name: str,
):
    config_path = os.path.join(dataset_dir, "dataset_config.json")
    try:
        with open(config_path, "r") as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset config not found at {config_path}")
    
    dataset_config = DatasetConfig(dataset_name=dataset_name, file_name=dataset_info[dataset_name]["file_name"])
    dataset_config.formatting = dataset_info[dataset_name].get("formatting", "sharegpt")
    dataset_config.common = CommonColumns(**dataset_info[dataset_name]["columns"])
    if "tags" in dataset_info[dataset_name]:  # only ShareGPT for now
        dataset_config.sharegpt = ShareGPTTags(**dataset_info[dataset_name]["tags"])

    return dataset_config