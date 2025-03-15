import pytest
from unittest.mock import patch, MagicMock
import torch
from transformers import PreTrainedTokenizerBase

from llavapool.data.collator import SFTDataCollatorWithPadAttentionMask

class TestCollator:
    """Tests for classes in collator.py"""
    
    def test_sft_data_collator_initialization(self):
        """Test initialization of SFTDataCollatorWithPadAttentionMask class"""
        # Create mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mock_tokenizer.pad_token_id = 0
        
        # Initialize class
        collator = SFTDataCollatorWithPadAttentionMask(
            tokenizer=mock_tokenizer,
            ignore_index=-100
        )
        
        # Verify initialization
        assert collator.tokenizer == mock_tokenizer
        assert collator.ignore_index == -100
    
    def test_collator_call_text_only(self):
        """Test collator call for text-only data"""
        # Create mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mock_tokenizer.pad_token_id = 0
        
        # Create mock input data
        features = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 1])
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "labels": torch.tensor([5, 6]),
                "attention_mask": torch.tensor([1, 1])
            }
        ]
        
        # Initialize and call class
        collator = SFTDataCollatorWithPadAttentionMask(
            tokenizer=mock_tokenizer,
            ignore_index=-100
        )
        result = collator(features)
        
        # Verify results
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (2, 4)  # Batch size 2, max length 4
        assert result["labels"].shape == (2, 4)
        assert result["attention_mask"].shape == (2, 4)
        
        # Verify padding
        assert result["input_ids"][1, 2] == 0  # Padding for second sample
        assert result["input_ids"][1, 3] == 0
        assert result["labels"][1, 2] == -100  # Padding with ignore_index
        assert result["labels"][1, 3] == -100
        assert result["attention_mask"][1, 2] == 0  # Padding part of attention_mask is 0
        assert result["attention_mask"][1, 3] == 0
    
    def test_collator_with_4d_attention_mask(self):
        """Test collator call for data with 4D attention_mask"""
        # Create mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mock_tokenizer.pad_token_id = 0
        
        # Create mock input data (including 4D attention_mask)
        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "attention_mask": torch.ones((1, 1, 3, 3))  # 4D tensor
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "attention_mask": torch.ones((1, 1, 2, 2))  # 4D tensor
            }
        ]
        
        # Initialize and call class
        collator = SFTDataCollatorWithPadAttentionMask(
            tokenizer=mock_tokenizer,
            ignore_index=-100
        )
        result = collator(features)
        
        # Verify results
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (2, 3)  # Batch size 2, max length 3
        assert result["labels"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 1, 3, 3)  # 4D attention_mask
        
        # Verify padding
        assert result["input_ids"][1, 2] == 0  # Padding for second sample
        assert result["labels"][1, 2] == -100  # Padding with ignore_index
    
    def test_collator_with_additional_fields(self):
        """Test collator call for data with additional fields"""
        # Create mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mock_tokenizer.pad_token_id = 0
        
        # Create mock input data (including additional fields)
        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "position_ids": torch.tensor([0, 1, 2]),
                "token_type_ids": torch.tensor([0, 0, 0])
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "position_ids": torch.tensor([0, 1]),
                "token_type_ids": torch.tensor([0, 0])
            }
        ]
        
        # Initialize and call class
        collator = SFTDataCollatorWithPadAttentionMask(
            tokenizer=mock_tokenizer,
            ignore_index=-100
        )
        result = collator(features)
        
        # Verify results
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert "position_ids" in result
        assert "token_type_ids" in result
        
        # Verify padding for additional fields
        assert result["position_ids"].shape == (2, 3)
        assert result["token_type_ids"].shape == (2, 3)
        assert result["position_ids"][1, 2] == 0  # Padding
        assert result["token_type_ids"][1, 2] == 0  # Padding
