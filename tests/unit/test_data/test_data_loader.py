import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import json
from argparse import Namespace

from llavapool.data.data_loader import convert_sharegpt, load_dataset
from llavapool.data.data_loader import load_dataset_module


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


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
        
        # Create mock DatasetConfig object with proper structure
        mock_config = MagicMock()
        mock_config.formatting = "sharegpt"
        
        # Set up common attributes
        mock_config.common.messages = "conversations"
        mock_config.common.images = None
        mock_config.common.videos = None
        mock_config.common.system = None
        mock_config.common.tools = None
        
        # Set up sharegpt attributes
        mock_config.sharegpt.role_tag = "from"
        mock_config.sharegpt.content_tag = "value"
        mock_config.sharegpt.user_tag = "human"
        mock_config.sharegpt.assistant_tag = "gpt"
        mock_config.sharegpt.system_tag = None
        mock_config.sharegpt.function_tag = "function"
        mock_config.sharegpt.observation_tag = "observation"
        
        # Call the function
        result = convert_sharegpt(
            example=mock_text_data,
            dataset_config=mock_config,
            data_args=Namespace(**{"dataset_dir": os.path.join(PROJECT_ROOT, "data")})
        )
        
        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert isinstance(result["_prompt"], list)
        assert isinstance(result["_response"], list)
        assert len(result["_prompt"]) == 1
        assert result["_prompt"][0]["content"] == "What is the capital of France?"
        assert result["_response"][0]["content"] == "The capital of France is Paris."
        assert result["_images"] is None
        assert result["_videos"] is None
    
    def test_convert_sharegpt_with_image(self):
        """Test for data conversion with image using real image from demo_data"""
        # Get the project root directory
        
        # Prepare mock data with real image path
        mock_image_data = {
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is in this image?"},
                {"from": "gpt", "value": "This is an image of the Eiffel Tower."}
            ],
            "images": ["demo_data/COCO_train2014_000000222016.jpg"]
        }
        
        # Create mock DatasetConfig object with proper structure
        mock_config = MagicMock()
        mock_config.formatting = "sharegpt"
        
        # Set up common attributes
        mock_config.common.messages = "conversations"
        mock_config.common.images = "images"
        mock_config.common.videos = None
        mock_config.common.system = None
        mock_config.common.tools = None
        
        # Set up sharegpt attributes
        mock_config.sharegpt.role_tag = "from"
        mock_config.sharegpt.content_tag = "value"
        mock_config.sharegpt.user_tag = "human"
        mock_config.sharegpt.assistant_tag = "gpt"
        mock_config.sharegpt.system_tag = None
        mock_config.sharegpt.function_tag = "function"
        mock_config.sharegpt.observation_tag = "observation"
        
        # Call the function with real data path
        result = convert_sharegpt(
            example=mock_image_data,
            dataset_config=mock_config,
            data_args=Namespace(**{"dataset_dir": os.path.join(PROJECT_ROOT, "data")})
        )
        
        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert "_images" in result
        assert isinstance(result["_images"], list)
        assert len(result["_images"]) == 1
        assert "COCO_train2014_000000222016.jpg" in result["_images"][0]
        assert "<image>" in result["_prompt"][0]["content"]
        assert "What is in this image?" in result["_prompt"][0]["content"]
        assert "This is an image of the Eiffel Tower." == result["_response"][0]["content"]
    
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
        
        # Create mock DatasetConfig object with proper structure
        mock_config = MagicMock()
        mock_config.formatting = "sharegpt"
        
        # Set up common attributes
        mock_config.common.messages = "conversations"
        mock_config.common.images = None
        mock_config.common.videos = None
        mock_config.common.system = None
        mock_config.common.tools = None

        # Set up sharegpt attributes
        mock_config.sharegpt.role_tag = "from"
        mock_config.sharegpt.content_tag = "value"
        mock_config.sharegpt.user_tag = "human"
        mock_config.sharegpt.assistant_tag = "gpt"
        mock_config.sharegpt.system_tag = None
        mock_config.sharegpt.function_tag = "function"
        mock_config.sharegpt.observation_tag = "observation"
        
        # Call the function
        result = convert_sharegpt(
            example=mock_multi_turn_data,
            dataset_config=mock_config,
            data_args=Namespace(**{"dataset_dir": os.path.join(PROJECT_ROOT, "data")})
        )

        # Verify the result
        assert "_prompt" in result
        assert "_response" in result
        assert isinstance(result["_prompt"], list)
        assert isinstance(result["_response"], list)
        assert len(result["_prompt"]) == 3
        assert result["_prompt"][0]["content"] == "Hello, how are you?"
        assert result["_prompt"][1]["content"] == "I'm doing well, thank you! How can I help you today?"
        assert result["_prompt"][2]["content"] == "Tell me about Python."
        assert result["_response"][0]["content"] == "Python is a high-level programming language..."

    # Set up sharegpt attributes
    def test_load_dataset_module(self):
        """Test the load_dataset_module function"""
        # Mock necessary objects and dependencies
        mock_converter = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        
        # Mock data arguments
        mock_data_args = MagicMock()
        mock_data_args.dataset = ["demo"]
        mock_data_args.dataset_dir = os.path.join(PROJECT_ROOT, "data")
        mock_data_args.streaming = False
        mock_data_args.val_size = 0.2
        mock_data_args.preprocessing_num_workers = 1
        mock_data_args.overwrite_cache = False
        mock_data_args.preprocessing_batch_size = 32
        mock_data_args.buffer_size = 1000
        mock_data_args.tokenized_path = None
        
        # Mock model arguments
        mock_model_args = MagicMock()
        mock_model_args.cache_dir = None
        
        # Mock training arguments
        mock_training_args = MagicMock()
        mock_training_args.seed = 42
        mock_training_args.local_process_index = 0
        mock_training_args.should_save = True
        
        # Mock dataset config
        mock_dataset_config = MagicMock()
        mock_dataset_config.file_name = "demo.json"
        mock_dataset_config.split = "train"
        mock_dataset_config.formatting = "sharegpt"
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset_dict = MagicMock()
        mock_dataset_dict.__getitem__.side_effect = lambda x: mock_dataset if x in ["train", "validation"] else None
        
        # Set up patchers
        with patch("llavapool.data.data_loader.get_dataset_config", return_value=mock_dataset_config) as mock_get_config, \
                patch("llavapool.data.data_loader.load_dataset", return_value=mock_dataset) as mock_load, \
                patch("llavapool.data.data_loader.convert_dataset", return_value=mock_dataset) as mock_convert, \
                patch("llavapool.data.data_loader.get_superivsed_dataset", return_value=mock_dataset) as mock_get_sup, \
                patch("llavapool.data.data_loader.split_dataset", return_value=mock_dataset_dict) as mock_split, \
                patch("os.path.isfile", return_value=True) as mock_isfile:
            
            
            # Call the function
            result = load_dataset_module(
                converter=mock_converter,
                data_args=mock_data_args,
                model_args=mock_model_args,
                training_args=mock_training_args,
                tokenizer=mock_tokenizer,
                processor=mock_processor
            )
            
            # Verify calls
            mock_get_config.assert_called_once_with(mock_data_args.dataset_dir, "demo")
            mock_load.assert_called_once()
            mock_convert.assert_called_once_with(mock_dataset, mock_dataset_config, mock_data_args, mock_training_args, format="sharegpt")
            mock_get_sup.assert_called_once()
            mock_split.assert_called_once()
            
            import pdb;pdb.set_trace()
            # Verify result
            assert "train_dataset" in result
            assert "eval_dataset" in result
            assert result["train_dataset"] == mock_dataset
            assert result["eval_dataset"] == mock_dataset
