import json
import os

import pytest

from llavapool.data.dataset_config import CommonColumns, DatasetConfig, ShareGPTTags, get_dataset_config


class TestDatasetConfig:
    """Tests for DatasetConfig class and related functions"""
    
    def test_dataset_config_initialization(self):
        """Test initialization of DatasetConfig class"""
        config = DatasetConfig(
            dataset_name="test_dataset",
            file_name="test_file.json",
            formatting="sharegpt",
        )
        
        assert config.dataset_name == "test_dataset"
        assert config.file_name == "test_file.json"
        assert config.formatting == "sharegpt"
        assert config.subset is None
        assert config.split == "train"
        assert config.num_samples is None
    
    def test_dataset_config_with_optional_params(self):
        """Test initialization of DatasetConfig class with optional parameters"""
        config = DatasetConfig(
            dataset_name="test_dataset",
            file_name="test_file.json",
            formatting="sharegpt",
            subset="subset1",
            split="validation",
            num_samples=100
        )
        
        assert config.dataset_name == "test_dataset"
        assert config.file_name == "test_file.json"
        assert config.formatting == "sharegpt"
        assert config.subset == "subset1"
        assert config.split == "validation"
        assert config.num_samples == 100
    
    def test_get_dataset_config(self, tmp_path):
        """Test get_dataset_config function"""
        # Create temporary config file
        mock_config = {
            "test_dataset": {
                "file_name": "test_data.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "images": "images"
                },
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt"
                }
            }
        }
        
        config_path = tmp_path / "dataset_config.json"
        with open(config_path, "w") as f:
            json.dump(mock_config, f)
        
        # Set environment variable (save original value)
        original_env = os.environ.get("DATASET_CONFIG_PATH", None)
        os.environ["DATASET_CONFIG_PATH"] = str(config_path)
        
        try:
            # Test function
            config = get_dataset_config("test_dataset")
            
            assert config.dataset_name == "test_dataset"
            assert config.file_name == "test_data.json"
            assert config.formatting == "sharegpt"
        finally:
            # Restore environment variable
            if original_env is not None:
                os.environ["DATASET_CONFIG_PATH"] = original_env
            else:
                del os.environ["DATASET_CONFIG_PATH"]
    
    def test_get_dataset_config_nonexistent(self):
        """Test error handling for nonexistent dataset"""
        with pytest.raises(ValueError, match="Dataset .* not found"):
            get_dataset_config("nonexistent_dataset")
    
    def test_common_columns_defaults(self):
        """Test default values of CommonColumns class"""
        columns = CommonColumns()
        
        assert columns.system is None
        assert columns.tools is None
        assert columns.images is None
        assert columns.videos is None
        assert columns.messages == "conversations"
    
    def test_sharegpt_tags_defaults(self):
        """Test default values of ShareGPTTags class"""
        tags = ShareGPTTags()
        
        assert tags.role_tag == "from"
        assert tags.content_tag == "value"
        assert tags.user_tag == "human"
        assert tags.assistant_tag == "gpt"
        assert tags.observation_tag == "observation"
        assert tags.function_tag == "function_call"
        assert tags.system_tag == "system"
