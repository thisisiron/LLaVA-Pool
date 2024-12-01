"""Constants and enums used across the codebase."""

# Supported model types
MODEL_TYPES = ["phi", "qwen2-vl", "llama"]

# Token indices
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

# Basic tokens
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
EOT_TOKEN = "<|eot_id|>"
PHI_EOT_TOKEN = "<|end|>\n"

# Chat template tokens
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
START_HEADER_TOKEN = "<|start_header_id|>"
END_HEADER_TOKEN = "<|end_header_id|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

# System message
SYSTEM_MESSAGE = "You are a helpful assistant."