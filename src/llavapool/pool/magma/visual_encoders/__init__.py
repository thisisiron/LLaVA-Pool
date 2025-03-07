import torch
from .clip import CustomCLIP
from ..utils import check_local_file


def build_encoder(config, init_kwargs):
    import pdb;pdb.set_trace()
    if config.model_type == "clip_vision_model":
        init_kwargs["pretrained_model_name_or_path"] = config.vision_name_or_path
        model = CustomCLIP.from_pretrained(
            config=config,
            torch_dtype=torch.bfloat16,
            **init_kwargs,
        )
    else:
        raise NotImplementedError()

    return model
