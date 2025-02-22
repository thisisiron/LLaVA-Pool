from typing import Any, Dict, Optional, TypedDict
from transformers import AutoConfig, PretrainedConfig
import importlib
from omegaconf import OmegaConf

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..pool import (
    PROCESSOR_MAPPING_NAMES, 
    CONFIG_MAPPING_NAMES, 
    MODEL_MAPPING_NAMES,
    IMAGE_PROCESSOR_MAPPING_NAMES
)
from ..utils.logging import get_logger
from ..utils.misc import (
    count_parameters, 
    skip_check_imports, 
    try_download_model_from_other_hub
)
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model

logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, mixin), {})


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer_and_processor(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_auto_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)

    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, config, tokenizer, model_args)

    except Exception as e:
        logger.warning("Processor was not found: {}.".format(e))
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

        if config.model_type in PROCESSOR_MAPPING_NAMES:
            image_processor = image_processor_class_from_name(config.model_type)()
            processor = processor_class_from_name(config.model_type)(image_processor, tokenizer)
            patch_processor(processor, config, tokenizer, model_args)
    
    return {"tokenizer": tokenizer, "processor": processor}


def load_tokenizer(model_args: "ModelArguments") -> "PreTrainedTokenizer":
    init_kwargs = _get_init_kwargs(model_args)
    config = load_local_config(model_args)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.text_config.name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            config.text_config.name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")
    return tokenizer


def image_processor_class_from_name(model_name: str):
    class_name = IMAGE_PROCESSOR_MAPPING_NAMES[model_name]
    module = importlib.import_module(f"..pool.{model_name}", __package__)
    return getattr(module, class_name)

def processor_class_from_name(model_name: str):
    class_name = PROCESSOR_MAPPING_NAMES[model_name]
    module = importlib.import_module(f"..pool.{model_name}", __package__)
    return getattr(module, class_name)


def config_class_from_name(model_name: str):
    class_name = CONFIG_MAPPING_NAMES[model_name]
    module = importlib.import_module(f"..pool.{model_name}", __package__)
    return getattr(module, class_name)


def model_class_from_name(model_name: str):
    if MODEL_MAPPING_NAMES.get(model_name) is None:
        return None
    class_name = MODEL_MAPPING_NAMES[model_name]
    module = importlib.import_module(f"..pool.{model_name}", __package__)
    return getattr(module, class_name)


def load_processor(model_args: "ModelArguments", tokenizer: "PreTrainedTokenizer") -> "ProcessorMixin":
    init_kwargs = _get_init_kwargs(model_args)
    config = load_local_config(model_args)
    processor_class = processor_class_from_name(config.name_or_path)
    
    try:
        image_processor = AutoProcessor.from_pretrained(config.vision_config.vision_name_or_path, **init_kwargs)
        image_processor = getattr(image_processor, 'image_processor', image_processor)
        processor = processor_class(image_processor, tokenizer)
        patch_processor(processor, config, tokenizer, model_args)
    except Exception as e:
        logger.warning("Processor was not found: {}.".format(e))
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def load_config(model_args: "ModelArguments", stage="sft") -> "PretrainedConfig":
    if stage == "sft":
        return load_auto_config(model_args)


def load_local_config(model_args: "ModelArguments") -> "PretrainedConfig":
    if model_args.yaml_path is not None:
        yaml_path = model_args.yaml_path

    local_config = OmegaConf.load(yaml_path)
    config_dict = OmegaConf.to_container(local_config, resolve=True)
    config_dict["name_or_path"] = local_config.model_name_or_path
    config_class = config_class_from_name(local_config.model_name_or_path)

    return config_class(**config_dict)

    
def load_auto_config(model_args: "ModelArguments") -> "PretrainedConfig":
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
    stage="sft"
) -> "PreTrainedModel":
    if stage == "sft":
        return load_auto_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)


def load_auto_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))
    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        elif model_class_from_name(config.model_type) is not None:
            model = model_class_from_name(config.model_type).from_pretrained(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            else:
                load_class = AutoModelForCausalLM
                
            if model_args.train_from_scratch:
                model = load_class.from_config(config)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model


def build_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    init_kwargs = _get_init_kwargs(model_args)
    config = load_local_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False

    if model is None and not lazy_load:
        # init_kwargs["config"] = config
        # init_kwargs["pretrained_model_name_or_path"] = config.model_name_or_path
        model_class = model_class_from_name(config.model_name_or_path)
        model = model_class(config, init_kwargs, dtype=model_args.compute_dtype)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )
    print(model.language_model.dtype)
    print(model.vision_model.dtype)
    print(next(model.abstractor.parameters()).dtype)
    print('end')
    return model