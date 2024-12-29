import os
from peft.utils import ModulesToSaveWrapper


def get_cache_dir():
    DEFAULT_HF_HOME = "~/.cache/huggingface"
    cache_dir = os.environ.get("HF_HOME", DEFAULT_HF_HOME)

    return cache_dir


def check_local_file(model_name_or_path):
    cache_dir = get_cache_dir()
    file_name = os.path.join(
        cache_dir, f"models--{model_name_or_path.replace('/', '--')}"
    )
    local_files_only = os.path.exists(file_name)
    file_name = file_name if local_files_only else model_name_or_path
    return local_files_only, file_name


def unwrap_peft(layer):
    """ This function is designed for the purpose of checking dtype of model or fetching model configs. """
    if isinstance(layer, ModulesToSaveWrapper):
        return layer.original_module
    else:
        return layer