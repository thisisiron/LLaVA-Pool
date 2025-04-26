from unittest.mock import MagicMock, patch

import torch

from llavapool.train.trainer_utils import calculate_loss, evaluate_model


class TestTrainerUtils:
    """Tests for functions in trainer_utils.py"""
    
    def test_calculate_loss(self):
        """Test for calculate_loss function"""
        # Set up mock model and input
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(loss=torch.tensor(2.5))
        
        # Mock input data
        input_data = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        
        # Call the function
        loss = calculate_loss(mock_model, input_data)
        
        # Verify the result
        mock_model.assert_called_once()
        assert loss == 2.5
    
    @patch('llavapool.train.trainer_utils.tqdm')
    def test_evaluate_model(self, mock_tqdm):
        """Test for evaluate_model function"""
        # Set up mock model and dataloader
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(loss=torch.tensor(2.0))
        
        # Mock dataloader
        mock_batch1 = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        mock_batch2 = {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[4, 5, 6]])
        }
        mock_dataloader = [mock_batch1, mock_batch2]
        
        # Mock tqdm
        mock_tqdm.return_value = mock_dataloader
        
        # Call the function
        eval_loss = evaluate_model(mock_model, mock_dataloader)
        
        # Verify the result
        assert mock_model.eval.called
        assert mock_model.call_count == 2
        assert eval_loss == 2.0  # Average loss of two batches
