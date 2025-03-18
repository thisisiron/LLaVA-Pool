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

    def test_load_dataset_module(self):
        """Test for load_dataset_module function using Qwen2.5-VL-3B-Instruct model"""
        
        # Import required libraries
        import transformers
        from llavapool.data.converter import load_converter
        
        # Setup model arguments
        model_args = MagicMock()
        model_args.cache_dir = None
        model_args.model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        # Setup data arguments
        data_args = MagicMock()
        data_args.dataset_dir = os.path.join(PROJECT_ROOT, "data")
        data_args.dataset = ["demo"]  # List 형식으로 dataset 이름 지정
        data_args.template = "qwen2_vl"
        data_args.streaming = False
        data_args.preprocessing_num_workers = 1
        data_args.overwrite_cache = False
        data_args.preprocessing_batch_size = 1
        data_args.val_size = 0.2  # 검증 데이터 비율 지정
        data_args.buffer_size = 1000
        data_args.tokenized_path = None
        data_args.cutoff_len = 1024
        
        # Setup training arguments
        training_args = MagicMock()
        training_args.local_process_index = 0
        training_args.seed = 42
        training_args.should_save = False
        
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        # 실제 converter 로드
        converter = load_converter(processor, tokenizer, data_args)
        
        # Call the function under test
        from llavapool.data.data_loader import load_dataset_module
        result = load_dataset_module(
            converter=converter,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            tokenizer=tokenizer,
            processor=processor,
            stage="sft"
        )
            
