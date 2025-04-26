from unittest.mock import MagicMock, call, patch

import torch

from llavapool.train.sft.workflow import run_supervised_fine_tuning


class TestWorkflow:
    """Tests for functions in workflow.py"""
    
    @patch('llavapool.train.sft.workflow.load_dataset')
    @patch('llavapool.train.sft.workflow.load_tokenizer')
    @patch('llavapool.train.sft.workflow.load_model')
    @patch('llavapool.train.sft.workflow.SFTTrainer')
    @patch('llavapool.train.sft.workflow.get_dataset_config')
    def test_run_supervised_fine_tuning(self, mock_get_dataset_config, mock_sft_trainer, 
                                       mock_load_model, mock_load_tokenizer, mock_load_dataset):
        """Test run_supervised_fine_tuning function"""
        # Set up mock objects
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_dataset = MagicMock()
        mock_dataset_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_dataset_dict
        
        mock_trainer = MagicMock()
        mock_sft_trainer.return_value = mock_trainer
        
        mock_dataset_config = MagicMock()
        mock_get_dataset_config.return_value = mock_dataset_config
        
        # Set up mock arguments
        model_args = MagicMock()
        model_args.model_name_or_path = "test/model"
        model_args.trust_remote_code = True
        
        data_args = MagicMock()
        data_args.dataset_names = ["test_dataset"]
        data_args.data_path = "/path/to/data"
        
        training_args = MagicMock()
        training_args.output_dir = "/path/to/output"
        training_args.num_train_epochs = 3
        training_args.per_device_train_batch_size = 4
        training_args.gradient_accumulation_steps = 2
        
        # Call the function
        run_supervised_fine_tuning(model_args, data_args, training_args)
        
        # Verify function calls
        mock_get_dataset_config.assert_called_once_with("test_dataset")
        mock_load_tokenizer.assert_called_once_with(
            model_args.model_name_or_path, 
            trust_remote_code=model_args.trust_remote_code
        )
        mock_load_model.assert_called_once_with(
            model_args.model_name_or_path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_args.trust_remote_code
        )
        mock_load_dataset.assert_called_once()
        mock_sft_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    @patch('llavapool.train.sft.workflow.load_dataset')
    @patch('llavapool.train.sft.workflow.load_tokenizer')
    @patch('llavapool.train.sft.workflow.load_model')
    @patch('llavapool.train.sft.workflow.SFTTrainer')
    @patch('llavapool.train.sft.workflow.get_dataset_config')
    def test_run_supervised_fine_tuning_multiple_datasets(self, mock_get_dataset_config, mock_sft_trainer, 
                                                        mock_load_model, mock_load_tokenizer, mock_load_dataset):
        """Test run_supervised_fine_tuning function with multiple datasets"""
        # Set up mock objects
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_dataset1 = MagicMock()
        mock_dataset2 = MagicMock()
        mock_dataset_dict1 = {"train": mock_dataset1}
        mock_dataset_dict2 = {"train": mock_dataset2}
        mock_load_dataset.side_effect = [mock_dataset_dict1, mock_dataset_dict2]
        
        mock_trainer = MagicMock()
        mock_sft_trainer.return_value = mock_trainer
        
        mock_dataset_config1 = MagicMock()
        mock_dataset_config2 = MagicMock()
        mock_get_dataset_config.side_effect = [mock_dataset_config1, mock_dataset_config2]
        
        # Set up mock arguments
        model_args = MagicMock()
        model_args.model_name_or_path = "test/model"
        model_args.trust_remote_code = True
        
        data_args = MagicMock()
        data_args.dataset_names = ["dataset1", "dataset2"]
        data_args.data_path = "/path/to/data"
        
        training_args = MagicMock()
        training_args.output_dir = "/path/to/output"
        training_args.num_train_epochs = 3
        training_args.per_device_train_batch_size = 4
        training_args.gradient_accumulation_steps = 2
        
        # Call the function
        run_supervised_fine_tuning(model_args, data_args, training_args)
        
        # Verify function calls
        assert mock_get_dataset_config.call_count == 2
        mock_get_dataset_config.assert_has_calls([
            call("dataset1"),
            call("dataset2")
        ])
        mock_load_tokenizer.assert_called_once()
        mock_load_model.assert_called_once()
        assert mock_load_dataset.call_count == 2
        mock_sft_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()
