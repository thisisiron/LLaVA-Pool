import ast
import os
import pathlib

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
)

from liger_kernel.transformers import apply_liger_kernel_to_mllama
from llavapool.data.data import make_supervised_data_module
from llavapool.config.params import DataArguments, ModelArguments, TrainingArguments
from llavapool.train.trainer import Phi3VTrainer, LLamaVTrainer
from llavapool.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def find_target_linear_names(
    model,
    num_lora_modules=-1,
    lora_namespan_exclude=None,
    verbose=True,
    model_type="phi"
):
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
        
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    if model_type == "phi" and 'embed_tokens' in lora_namespan_exclude:
        lora_namespan_exclude.remove('embed_tokens')
        lora_namespan_exclude += ['model.embed_tokens']

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(
            f"Found {len(lora_module_names)} lora modules: {lora_module_names}"
        )
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(
    model,
    training_args,
    compute_dtype,
    device,
    model_type="phi"
):
    if model_type == "phi":
        vision_tower = model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(dtype=compute_dtype, device=device)
        img_projection_params = (
            model.vision_embed_tokens.img_projection.parameters()
        )
    else:  # llama
        vision_tower = model.vision_model
        vision_tower.to(dtype=compute_dtype, device=device)
        img_projection_params = model.multi_modal_projector.parameters()

    set_requires_grad(
        img_projection_params,
        training_args.tune_img_projector
    )
    vision_model_params = vision_tower.parameters()
    set_requires_grad(
        vision_model_params,
        not training_args.freeze_vision_tower
    )

    if training_args.bits in [4, 8]:
        if model_type == "phi":
            model.vision_embed_tokens.img_processor.to(
                dtype=compute_dtype,
                device=device
            )
        else:
            model.multi_modal_projector.to(
                dtype=compute_dtype,
                device=device
            )


def configure_llm(model, training_args, model_type="phi"):
    if model_type == "phi":
        lm_head_params = model.lm_head.parameters()
        set_requires_grad(lm_head_params, not training_args.freeze_llm)

        embed_token_params = model.model.embed_tokens.parameters()
        set_requires_grad(embed_token_params, not training_args.freeze_llm)

        for name, param in model.model.named_parameters():
            if name.startswith('layers') or name.startswith('norm'):
                param.requires_grad = not training_args.freeze_llm
    else:  # llama
        llm_params = model.language_model.parameters()
        set_requires_grad(llm_params, not training_args.freeze_llm)


def module_filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
    if fqn == "1":
        return False
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if model_args.model_type == "llama":
        apply_liger_kernel_to_mllama()

    if model_args.model_type == "phi":
        assert training_args.num_crops <= 16, (
            'num_crops must be less than or equal to 16'
        )
    
    assert not (training_args.lora_enable and training_args.freeze_llm), (
        'When using LoRA, the LLM should not be frozen. '
        'If you want to freeze the LLM, please disable LoRA.'
    )

    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "Error: training_args.lora_enable is not enabled, "
            "but training_args.vision_lora is enabled."
        )
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = (
                [] if model_args.model_type == "phi"
                else ["multi_modal_projector"]
            )

        if not training_args.vision_lora:
            excluded_modules = (
                ["vision_model", "img_projection"]
                if model_args.model_type == "phi"
                else ["vision_model", "multi_modal_projector"]
            )
            training_args.lora_namespan_exclude += excluded_modules

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        skip_modules = (
            ["img_projection", "vision_model"]
            if model_args.model_type == "phi"
            else ["multi_modal_projector", "vision_model"]
        )
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=skip_modules,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    # Initialize model
    if model_args.model_type == "phi":
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            _attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "eager"
            ),
            **bnb_model_from_pretrained_args
        )
        model.config.use_cache = False
    else:  # llama
        model = MllamaForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "sdpa"
            ),
            **bnb_model_from_pretrained_args
        )
        model.config.hidden_size = model.config.text_config.hidden_size
        model.config.text_config.use_cache = False

    # Configure quantization and gradient checkpointing
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Configure LoRA
    if training_args.lora_enable:
        target_modules = find_target_linear_names(
            model,
            lora_namespan_exclude=training_args.lora_namespan_exclude,
            num_lora_modules=training_args.num_lora_modules,
            model_type=model_args.model_type
        )
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM" if model_args.model_type == "phi" else None
        )
        
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
                
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    # Initialize processor
    if model_args.model_type == "phi":
        processor = AutoProcessor.from_pretrained(
            model_args.model_id,
            cache_dir=training_args.cache_dir,
            padding_side='right',
            trust_remote_code=True,
            num_crops=training_args.num_crops,
            model_max_length=training_args.max_seq_length
        )
        processor.tokenizer.pad_token = processor.tokenizer.unk_token
        processor.tokenizer.pad_token_id = (
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.pad_token
            )
        )
    else:  # llama
        processor = AutoProcessor.from_pretrained(model_args.model_id)

    processor.tokenizer.padding_side = 'right'

    # Update model config with tokenizer settings
    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    # Configure model components
    if training_args.lora_enable:
        model_to_configure = (
            model.model.model if model_args.model_type == "phi" else model.model
        )
    else:
        model_to_configure = (
            model.model if model_args.model_type == "phi" else model
        )
        configure_llm(model, training_args, model_args.model_type)

    if not training_args.vision_lora:
        configure_vision_tower(
            model_to_configure,
            training_args,
            compute_dtype,
            training_args.device,
            model_args.model_type
        )

    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    # Configure dtype for quantized models
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)

            if model_args.model_type == "phi":
                if ('lm_head' in name or 'embed_token' in name
                        and 'vision_embed_token' not in name):
                    if hasattr(module, 'weight'):
                        if (training_args.bf16
                                and module.weight.dtype == torch.float32):
                            module = module.to(torch.bfloat16)
            else:  # llama
                if 'lm_head' in name or 'embed_token' in name:
                    if hasattr(module, 'weight'):
                        if (training_args.bf16
                                and module.weight.dtype == torch.float32):
                            module = module.to(torch.bfloat16)

    # Set up data module and trainer
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        model_type=model_args.model_type,
    )

    trainer_class = (
        Phi3VTrainer if model_args.model_type == "phi" else LLamaVTrainer
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        **data_module
    }
    if model_args.model_type == "phi":
        trainer_kwargs["processor"] = processor

    trainer = trainer_class(**trainer_kwargs)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if model_args.model_type == "phi":
        model.config.use_cache = True
    else:  # llama
        model.config.text_config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(),
            training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(),
            require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir,
                state_dict=state_dict
            )
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin")
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer,
            output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()