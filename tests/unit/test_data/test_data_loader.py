import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import json

from llavapool.data.data_loader import convert_sharegpt, load_dataset

class TestDataLoader:
    """Tests for functions in data_loader.py"""
    
    def test_convert_sharegpt_text_only(self):
        """Test for text-only data conversion"""
        # Prepare mock data
        mock_text_data = {
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."}
            ]
        }
        
        # Create mock DatasetConfig object
        mock_config = MagicMock()
        mock_config.formatting = "sharegpt"
        mock_config.columns.messages = "conversations"
        mock_config.columns.images = "images"
        mock_config.tags.role_tag = "from"
        mock_config.tags.content_tag = "value"
        mock_config.tags.user_tag = "human"
        mock_config.tags.assistant_tag = "gpt"
        
        # Call the function
        result = convert_sharegpt(
            example=mock_text_data,
            dataset_config=mock_config,
            data_args={"data_path": "/dummy/path"}
        )
        
        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert isinstance(result["_prompt"], list)
        assert isinstance(result["_response"], list)
        assert len(result["_prompt"]) == 1
        assert "What is the capital of France?" in result["_prompt"][0]
        assert "The capital of France is Paris." in result["_response"][0]
        assert result["_images"] is None
        assert result["_videos"] is None
    
    def test_convert_sharegpt_with_image(self):
        """Test for data conversion with image using real image from demo_data"""
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        
        # Load demo.json to get real image data
        demo_json_path = os.path.join(project_root, "data", "demo.json")
        with open(demo_json_path, 'r') as f:
            demo_data = json.load(f)
        
        # Use the first example from demo.json
        real_example = {
            "conversations": [
                {"from": "human", "value": "<image>이미지에 대해 설명해줘."},
                {"from": "gpt", "value": "큰 빨간 공중 전화 박스 안에 한 남자가 서 있습니다."}
            ],
            "images": ["demo_data/COCO_train2014_000000222016.jpg"]
        }
        
        # Create mock DatasetConfig object
        mock_config = MagicMock()
        mock_config.formatting = "sharegpt"
        mock_config.columns.messages = "conversations"
        mock_config.columns.images = "images"
        mock_config.tags.role_tag = "from"
        mock_config.tags.content_tag = "value"
        mock_config.tags.user_tag = "human"
        mock_config.tags.assistant_tag = "gpt"
        
        # Call the function with real data path
        result = convert_sharegpt(
            example=real_example,
            dataset_config=mock_config,
            data_args={"dataset_dir": os.path.join(project_root, "data")}
        )
        
        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert "_images" in result
        assert isinstance(result["_images"], list)
        assert len(result["_images"]) == 1
        assert "COCO_train2014_000000222016.jpg" in result["_images"][0]
        assert "<image>" in result["_prompt"][0]
        assert "이미지에 대해 설명해줘." in result["_prompt"][0]
        assert "큰 빨간 공중 전화 박스 안에 한 남자가 서 있습니다." in result["_response"][0]
    
    def test_convert_sharegpt_multi_turn(self):
        """Test for multi-turn conversation conversion"""
        # Prepare mock data
        mock_multi_turn_data = {
            "conversations": [
                {"from": "human", "value": "Hello, how are you?"},
                {"from": "gpt", "value": "I'm doing well, thank you! How can I help you today?"},
                {"from": "human", "value": "Tell me about Python."},
                {"from": "gpt", "value": "Python is a high-level programming language..."}
            ]
        }
        
        # Create mock DatasetConfig object
        mock_config = MagicMock()
        mock_config.formatting = "sharegpt"
        mock_config.columns.messages = "conversations"
        mock_config.columns.images = "images"
        mock_config.tags.role_tag = "from"
        mock_config.tags.content_tag = "value"
        mock_config.tags.user_tag = "human"
        mock_config.tags.assistant_tag = "gpt"
        
        # Call the function
        result = convert_sharegpt(
            example=mock_multi_turn_data,
            dataset_config=mock_config,
            data_args={"data_path": "/dummy/path"}
        )
        
        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert isinstance(result["_prompt"], list)
        assert isinstance(result["_response"], list)
        assert len(result["_prompt"]) == 2
        assert "Hello, how are you?" in result["_prompt"][0]
        assert "I'm doing well, thank you! How can I help you today?" in result["_prompt"][1]
        assert "Tell me about Python." in result["_prompt"][1]
        assert "Python is a high-level programming language..." in result["_response"][1]
    
    @patch('llavapool.data.data_loader.DatasetDict')
    @patch('llavapool.data.data_loader.load_dataset')
    def test_load_dataset(self, mock_load_dataset, mock_dataset_dict):
        """Test for load_dataset function"""
        # Set up mock objects
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset_dict.return_value = {"train": mock_dataset}
        
        # Create mock DatasetConfig object
        mock_config = MagicMock()
        mock_config.dataset_name = "test_dataset"
        mock_config.file_name = "test_file.json"
        mock_config.formatting = "sharegpt"
        mock_config.split = "train"
        mock_config.subset = None
        mock_config.num_samples = None
        
        # Call the function
        result = load_dataset(mock_config, data_path="/dummy/path")
        
        # Verify the function call
        mock_load_dataset.assert_called_once()
        assert result == {"train": mock_dataset}
    
    @patch('llavapool.data.data_loader.DatasetDict')
    @patch('llavapool.data.data_loader.load_dataset')
    def test_load_dataset_with_subset(self, mock_load_dataset, mock_dataset_dict):
        """Test for load_dataset function with subset"""
        # Set up mock objects
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset_dict.return_value = {"train": mock_dataset}
        
        # Create mock DatasetConfig object
        mock_config = MagicMock()
        mock_config.dataset_name = "test_dataset"
        mock_config.file_name = "test_file.json"
        mock_config.formatting = "sharegpt"
        mock_config.split = "train"
        mock_config.subset = "subset1"
        mock_config.num_samples = None
        
        # Call the function
        result = load_dataset(mock_config, data_path="/dummy/path")
        
        # Verify the function call
        mock_load_dataset.assert_called_once()
        assert result == {"train": mock_dataset}
    
    @patch('llavapool.data.data_loader.DatasetDict')
    @patch('llavapool.data.data_loader.load_dataset')
    def test_load_dataset_with_num_samples(self, mock_load_dataset, mock_dataset_dict):
        """Test for load_dataset function with num_samples"""
        # Set up mock objects
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        mock_dataset_dict.return_value = {"train": mock_dataset}
        
        # Create mock DatasetConfig object
        mock_config = MagicMock()
        mock_config.dataset_name = "test_dataset"
        mock_config.file_name = "test_file.json"
        mock_config.formatting = "sharegpt"
        mock_config.split = "train"
        mock_config.subset = None
        mock_config.num_samples = 100
        
        # Call the function
        result = load_dataset(mock_config, data_path="/dummy/path")
        
        # Verify the function call
        mock_load_dataset.assert_called_once()
        mock_dataset.select.assert_called_once()
        assert result == {"train": mock_dataset}
