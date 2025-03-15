import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

from llavapool.data.template import get_template_by_name, apply_template

class TestTemplate:
    """Tests for functions in template.py"""
    
    @patch('llavapool.data.template.importlib.import_module')
    def test_get_template_by_name_valid(self, mock_import_module):
        """Test get_template_by_name function with a valid template name"""
        # Set up mock template module
        mock_template = MagicMock()
        mock_import_module.return_value = mock_template
        
        # Call the function
        result = get_template_by_name("llava")
        
        # Verify function call
        mock_import_module.assert_called_once_with("llavapool.data.template.llava")
        assert result == mock_template
    
    def test_get_template_by_name_invalid(self):
        """Test get_template_by_name function with an invalid template name"""
        with pytest.raises(ImportError):
            get_template_by_name("nonexistent_template")
    
    def test_apply_template_text_only(self):
        """Test applying template to text-only conversation"""
        # Set up mock template
        mock_template = MagicMock()
        mock_template.roles = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System"
        }
        mock_template.messages_to_prompt = MagicMock(return_value="User: Hello\nAssistant: Hi")
        
        # Prepare mock messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        # Call the function
        result = apply_template(mock_template, messages)
        
        # Verify result
        assert result == "User: Hello\nAssistant: Hi"
        mock_template.messages_to_prompt.assert_called_once_with(messages)
    
    def test_apply_template_with_system(self):
        """Test applying template to conversation with system message"""
        # Set up mock template
        mock_template = MagicMock()
        mock_template.roles = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System"
        }
        mock_template.messages_to_prompt = MagicMock(
            return_value="System: Be helpful\nUser: Hello\nAssistant: Hi"
        )
        
        # Prepare mock messages
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        # Call the function
        result = apply_template(mock_template, messages)
        
        # Verify result
        assert result == "System: Be helpful\nUser: Hello\nAssistant: Hi"
        mock_template.messages_to_prompt.assert_called_once_with(messages)
    
    def test_apply_template_with_image(self):
        """Test applying template to conversation with image"""
        # Set up mock template
        mock_template = MagicMock()
        mock_template.roles = {
            "user": "User",
            "assistant": "Assistant"
        }
        mock_template.messages_to_prompt = MagicMock(
            return_value="User: <image>\nWhat's in this image?\nAssistant: A cat."
        )
        
        # Prepare mock messages
        messages = [
            {"role": "user", "content": "<image>\nWhat's in this image?"},
            {"role": "assistant", "content": "A cat."}
        ]
        
        # Call the function
        result = apply_template(mock_template, messages)
        
        # Verify result
        assert result == "User: <image>\nWhat's in this image?\nAssistant: A cat."
        mock_template.messages_to_prompt.assert_called_once_with(messages)
