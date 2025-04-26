from unittest.mock import MagicMock, patch

import torch

from llavapool.model.model_loader import load_model, load_processor, load_tokenizer, load_tokenizer_and_processor


class TestModelLoader:
    """Tests for functions in model_loader.py"""
    
    @patch('llavapool.model.model_loader.AutoTokenizer')
    @patch('llavapool.model.model_loader.AutoProcessor')
    def test_load_tokenizer_and_processor(self, mock_auto_processor, mock_auto_tokenizer):
        """Test for load_tokenizer_and_processor function"""
        # Set up mock objects
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_processor.from_pretrained.return_value = mock_processor
        
        # Call the function
        tokenizer, processor = load_tokenizer_and_processor(
            model_name_or_path="test/model",
            trust_remote_code=True
        )
        
        # Verify the function call
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "test/model", trust_remote_code=True
        )
        mock_auto_processor.from_pretrained.assert_called_once_with(
            "test/model", trust_remote_code=True
        )
        assert tokenizer == mock_tokenizer
        assert processor == mock_processor
    
    @patch('llavapool.model.model_loader.AutoTokenizer')
    def test_load_tokenizer(self, mock_auto_tokenizer):
        """Test for load_tokenizer function"""
        # Set up mock objects
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Call the function
        tokenizer = load_tokenizer(
            model_name_or_path="test/model",
            trust_remote_code=True
        )
        
        # Verify the function call
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "test/model", trust_remote_code=True
        )
        assert tokenizer == mock_tokenizer
    
    @patch('llavapool.model.model_loader.AutoProcessor')
    def test_load_processor(self, mock_auto_processor):
        """Test for load_processor function"""
        # Set up mock objects
        mock_processor = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor
        
        # Call the function
        processor = load_processor(
            model_name_or_path="test/model",
            trust_remote_code=True
        )
        
        # Verify the function call
        mock_auto_processor.from_pretrained.assert_called_once_with(
            "test/model", trust_remote_code=True
        )
        assert processor == mock_processor
    
    @patch('llavapool.model.model_loader.AutoModelForCausalLM')
    def test_load_model(self, mock_auto_model):
        """Test for load_model function"""
        # Set up mock objects
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Call the function
        model = load_model(
            model_name_or_path="test/model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Verify the function call
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test/model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        assert model == mock_model
    
    @patch('llavapool.model.model_loader.AutoModelForCausalLM')
    def test_load_model_with_low_cpu_mem_usage(self, mock_auto_model):
        """Test for load_model function with low_cpu_mem_usage option"""
        # Set up mock objects
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Call the function
        model = load_model(
            model_name_or_path="test/model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Verify the function call
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test/model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        assert model == mock_model
    
    @patch('llavapool.model.model_loader.AutoModelForCausalLM')
    def test_load_model_with_additional_kwargs(self, mock_auto_model):
        """Test for load_model function with additional kwargs"""
        # Set up mock objects
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Call the function
        model = load_model(
            model_name_or_path="test/model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="flash_attention_2"
        )
        
        # Verify the function call
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test/model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="flash_attention_2"
        )
        assert model == mock_model
