import torch
from .clip import CustomCLIP
from ..utils import check_local_file


def build_encoder(config, init_kwargs):
    if config.vision_encoder_type == "openai.clip":
        init_kwargs["pretrained_model_name_or_path"] = config.vision_name_or_path
        model = CustomCLIP.from_pretrained(
            config=config,
            torch_dtype=torch.bfloat16,
            **init_kwargs,
        )
    else:
        raise NotImplementedError()

    return model