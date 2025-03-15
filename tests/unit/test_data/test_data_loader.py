import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

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
        """Test for data conversion with image"""
        # Prepare mock data
        mock_image_data = {
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is in this image?"},
                {"from": "gpt", "value": "This is an image of the Eiffel Tower."}
            ],
            "images": ["path/to/image.jpg"]
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
            example=mock_image_data,
            dataset_config=mock_config,
            data_args={"data_path": "/dummy/path"}
        )
        
        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert "_images" in result
        assert isinstance(result["_images"], list)
        assert len(result["_images"]) == 1
        assert result["_images"][0] == "path/to/image.jpg"
        assert "<image>" in result["_prompt"][0]
        assert "What is in this image?" in result["_prompt"][0]
        assert "This is an image of the Eiffel Tower." in result["_response"][0]
    
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
