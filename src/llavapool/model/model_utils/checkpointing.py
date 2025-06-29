import inspect
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch

from ...utils.constants import LAYERNORM_NAMES
from ...utils.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def get_unsloth_gradient_checkpointing_func() -> Callable:
    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""
        Saves VRAM by smartly offloading to RAM.
        """

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            with torch.no_grad():
                output = forward_function(hidden_states, *args)

            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            return output

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            (hidden_states,) = ctx.saved_tensors
            hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
            hidden_states.requires_grad_(True)
            with torch.enable_grad():
                (output,) = ctx.forward_function(hidden_states, *ctx.args)

            torch.autograd.backward(output, grad_output)
            return (None, hidden_states.grad) + (None,) * len(ctx.args)

    return UnslothGradientCheckpointing.apply


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""Only applies gradient checkpointing to trainable layers."""

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        if isinstance(func, partial):
            module: torch.nn.Module = func.func.__self__
        else:
            module: torch.nn.Module = func.__self__

        has_grad = False
        if any(param.requires_grad for param in module.parameters()):
            has_grad = True
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
                    break  # assume the first tensor is always the hidden states

        if has_grad:
            return gradient_checkpointing_func(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""
    Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    if use_unsloth_gc:
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) add the upcasting of the lm_head in fp32
    """
    if model_args.upcast_layernorm:
        logger.info("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info("Gradient checkpointing enabled.")

    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info("Upcasting lm_head outputs in float32.")
            output_layer.register_forward_hook(_fp32_forward_post_hook)
