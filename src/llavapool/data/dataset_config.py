import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CommonColumns:
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    messages: str = "conversations"

    # dpo columns
    chosen: Optional[str] = "chosen"
    rejected: Optional[str] = "rejected"
    kto_tag: Optional[str] = None


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
    preference: bool = False
    
    # Extra configs
    subset: Optional[str] = None
    split: str = field(default="train")
    num_samples: Optional[int] = None
    
    # Nested dataclasses
    common: CommonColumns = field(default_factory=CommonColumns)
    sharegpt: Optional[ShareGPTTags] = field(default=ShareGPTTags)
    
    def __repr__(self) -> str:
        """Return dataset name as string representation of the object"""
        return self.dataset_name
    
    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        """Set an attribute from a dictionary with a given key
        
        Args:
            key: Name of the attribute to set
            obj: Dictionary to get value from
            default: Default value if key is not in dictionary
        """
        setattr(self, key, obj.get(key, default))
    
    def join(self, attr: Dict[str, Any]) -> None:
        """Set multiple attributes at once from a dictionary
        
        Args:
            attr: Dictionary containing attributes to set
        """
        # Basic settings
        self.set_attr("formatting", attr, default="sharegpt")
        self.set_attr("preference", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("num_samples", attr)
        
        # Nested CommonColumns settings
        if "columns" in attr:
            column_dict = attr["columns"]
            if isinstance(column_dict, dict):
                self.common = CommonColumns(**column_dict)
        
        # Nested ShareGPTTags settings
        if "tags" in attr:
            tags_dict = attr["tags"]
            if isinstance(tags_dict, dict):
                self.sharegpt = ShareGPTTags(**tags_dict)


def get_dataset_config(
    dataset_dir: str,
    dataset_name: str,
):
    """Load settings from the dataset config file and return a DatasetConfig object
    
    Args:
        dataset_dir: Directory containing the dataset config file
        dataset_name: Name of the dataset to load
    
    Returns:
        DatasetConfig: Configured dataset settings object
    """
    config_path = os.path.join(dataset_dir, "dataset_config.json")
    try:
        with open(config_path, "r") as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset config not found at {config_path}")

    # TODO: dataset_name should be a list. We need to handle this case.
    dataset_config = DatasetConfig(
        dataset_name=dataset_name, 
        file_name=dataset_info[dataset_name]["file_name"]
    )
    
    # Apply all settings at once using the join method
    dataset_config.join(dataset_info[dataset_name])

    return dataset_config