"""Parameter classes for model training."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of the model (phi, qwen, llama)"}
    )
    # Define acceptable choices
    allowed_model_ids: set = field(
        default_factory=lambda: {
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-2-7B-Instruct",
            "meta-llama/Llama-1.5-3B-Vision",
            "microsoft/Phi-3.5-vision-instruct",
            "microsoft/Phi-3-vision-128k-instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
        },
        metadata={"help": "Set of allowed model IDs"}
    )

    def __post_init__(self):
        """Validate model_id after initialization."""
        if self.model_id not in self.allowed_model_ids:
            raise ValueError(
                f"Invalid model_id. Choose from {self.allowed_model_ids}"
            )


@dataclass
class TrainingArguments(HfTrainingArguments):
    """Arguments pertaining to training configuration."""

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for caching model files"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use for training"}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 parameter for Adam optimizer"}
    )
    adam_beta2: float = field(
        default=0.98,
        metadata={"help": "Beta2 parameter for Adam optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-7,
        metadata={"help": "Epsilon parameter for Adam optimizer"}
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Whether to freeze vision tower parameters"}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze LLM parameters"}
    )
    tune_img_projector: bool = field(
        default=True,
        metadata={"help": "Whether to tune image projector"}
    )
    disable_flash_attn2: bool = field(
        default=True,
        metadata={"help": "Whether to disable Flash Attention 2"}
    )
    max_seq_length: int = field(
        default=131072,  # Default for phi3-vision-128k-instruct
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded "
                   "(and possibly truncated)."
        }
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress quantization statistics through double quantization"
        }
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`"
        }
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use for quantization"}
    )
    lora_enable: bool = field(
        default=False,
        metadata={"help": "Whether to enable LoRA"}
    )
    vision_lora: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to vision tower"}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA matrices"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": "Path to pretrained LoRA weights"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Type of bias to use in LoRA"}
    )
    projector_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for projector"}
    )
    vision_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for vision tower"}
    )
    lora_namespan_exclude: str = field(
        default=None,
        metadata={"help": "List of namespan to exclude for LoRA"}
    )
    num_lora_modules: int = field(
        default=-1,
        metadata={"help": "Number of LoRA modules to use"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "Number of crops for vision preprocessing"}
    )
    tune_merger: bool = field(
        default=False,
        metadata={"help": "Whether to tune merger module"}
    )
    merger_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for merger module"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to data processing."""

    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data"}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy preprocessing"}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder containing images"}
    )
    max_num_frames: int = field(
        default=10,
        metadata={"help": "Maximum number of frames for video processing"}
    )
    min_pixels: int = field(
        default=512 * 28 * 28,
        metadata={"help": "Minimum number of pixels for image processing"}
    )
    max_pixels: int = field(
        default=1280 * 28 * 28,
        metadata={"help": "Maximum number of pixels for image processing"}
    )
    fps: float = field(
        default=1.0,
        metadata={"help": "Frames per second for video processing"}
    )