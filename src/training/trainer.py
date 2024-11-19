import os
from typing import Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    PREFIX_CHECKPOINT_DIR,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    get_parameter_names,
    is_peft_available,
    is_sagemaker_mp_enabled,
    logger,
)

from training.train_utils import (
    maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)


class BaseVTrainer(Trainer):
    """공통 기능을 가진 기본 트레이너 클래스"""
    
    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name
            ]
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper[self.projector_key] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr
            
            if len(lr_mapper) > 0:
                special_lr_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if any(module_keyword in name for module_keyword in lr_mapper)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and 
                                n not in special_lr_parameters and 
                                p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and 
                                n not in special_lr_parameters and 
                                p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [
                        name for name, _ in opt_model.named_parameters()
                        if module_keyword in name
                    ]
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and 
                                    n in module_parameters and 
                                    p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and 
                                    n in module_parameters and 
                                    p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {p.data_ptr(): p.numel() 
                             for p in module.parameters()}.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.lora_enable:
            checkpoint_folder = (
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=False
            )
            torch.save(
                non_lora_weights,
                os.path.join(output_dir, "non_lora_state_dict.bin")
            )

            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(output_dir)
                self._save_rng_state(output_dir)

            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            if self.args.should_save:
                self.state.stateful_callbacks["TrainerControl"] = (
                    self.control.state()
                )
                self.state.save_to_json(
                    os.path.join(output_dir, TRAINER_STATE_NAME)
                )

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
        else:
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = ((PreTrainedModel,) if not is_peft_available() 
                           else (PreTrainedModel, PeftModel))
        
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), 
                        supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, "
                    "only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            state_dict = {k: v for k, v in state_dict.items() 
                         if "wte" not in k}
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class LLamaVTrainer(BaseVTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector_key = "multi_modal_projector"


class Phi3VTrainer(BaseVTrainer):
    def __init__(self, *args, processor: Optional[ProcessorMixin] = None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.processor.chat_template = None
        self.projector_key = "img_projection"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
