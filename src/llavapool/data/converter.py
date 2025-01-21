import re
import math
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Union, Sequence
from typing_extensions import override

from decord import VideoReader
from PIL import Image
import numpy as np
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)

from ..utils.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from .template import get_template_and_fix_tokenizer
from .data_loader import Role


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def basic_resize(image, image_resolution: int = 512):
    if max(image.width, image.height) > image_resolution:
        resize_factor = image_resolution / max(image.width, image.height)
        resized_width = int(image.width * resize_factor)
        resized_height = int(image.height * resize_factor)
        image = image.resize((resized_width, resized_height))
    return image


class BaseConverter:
    def __init__(self, template, processor, tokenizer):
        self.template = template
        self.processor = processor
        self.tokenizer = tokenizer
        self.expand_mm_tokens = True

        self.image_token = template.image_token
        self.video_token = template.video_token

    def _check_input(self, images, videos):
        if images is not None and len(images) >= 1 and self.image_token is None:
            raise ValueError("Image token is required for images")
        if videos is not None and len(videos) >= 1 and self.video_token is None:
            raise ValueError("Video token is required for videos")

    def convert(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def __call__(self):
        return self.convert()

    def _preprocess_image(self, image, **kwargs):
        image = image.convert("RGB")
        image = basic_resize(image, image_resolution=kwargs.get("image_resolution", 640))
        return image

    def _get_images(self, images, **kwargs):
        preprocessed_images = []

        for image in images:
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            
            if pil_image is None:
                raise ValueError("Invalid image type")
            
            pil_image = self._preprocess_image(pil_image, **kwargs)
            preprocessed_images.append(pil_image)

        return preprocessed_images
    
    def _get_videos(self, videos, **kwargs):
        preprocessed_videos = []
        for video in videos:
            if isinstance(video, str):
                vr = VideoReader(video)
                total_frames = len(vr)
                video_fps = vr.get_avg_fps()
                sample_frames = self._get_sample_frames(total_frames, video_fps, **kwargs)
                frame_indices = np.linspace(0, total_frames - 1, sample_frames).astype(int)
                frames = vr.get_batch(frame_indices).asnumpy()
            else:
                raise ValueError("Invalid video type")
            
            frames = self._get_images(frames, **kwargs)
            preprocessed_videos.append(frames)

        return preprocessed_videos
    
    def _get_sample_frames(self, total_frames, video_fps, **kwargs):
        assert not ("fps" in kwargs and "nframes" in kwargs), "Cannot specify both `fps` and `max_num_frames`"

        if "fps" in kwargs:
            fps = kwargs.get("fps")
            min_frames = ceil_by_factor(kwargs.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(kwargs.get("max_frames", FPS_MAX_FRAMES), FRAME_FACTOR)
            sample_frames = total_frames / video_fps * fps
            sample_frames = min(max_frames, max(min_frames, sample_frames))
            sample_frames = round_by_factor(sample_frames, FRAME_FACTOR)
        elif "nframes" in kwargs:
            sample_frames = round_by_factor(kwargs.get("nframes"), FRAME_FACTOR)

        if FRAME_FACTOR > sample_frames:
            raise ValueError("Number of frames is too small")
        if sample_frames > total_frames:
            raise ValueError("Number of frames is too large")

        return sample_frames
    
    def get_visual_inputs(self, images, videos, batch_ids=None, seq_lens=None, **kwargs):
        image_processor = self.processor.image_processor
        video_processor = getattr(self.processor, 'video_processor', None) or image_processor

        image_resolution = getattr(self.processor, "image_resolution", 640)
        input_dict = {"images": None}

        if len(images) != 0:
            input_dict["images"] = self._get_images(
                images, 
                image_resolution=image_resolution,
                **kwargs
            )

        if len(videos) != 0:
            input_dict["videos"] = self._get_videos(
                videos, 
                **kwargs
        )

        visual_inputs = {}
        if input_dict.get("images") is not None:
            visual_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
        if input_dict.get("videos") is not None:
            visual_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        return visual_inputs

    def process_media_tokens(self, messages, images, videos):
        # raise NotImplementedError("This method must be implemented in the subclass")
        self._check_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    def encode_multi_turn(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence[str],
        videos: Sequence[str],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        messages = self.process_media_tokens(messages, images, videos)
        encoded_messages = self._encode(messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def _replace_tokens(self, template: str, **kwargs) -> str:
        result = template

        for key, value in kwargs.items():
            result = result.replace("{{" + key + "}}", str(value))

        token_pattern = r'\{\{(\w+_token)\}\}'  # ex. bos_token, eos_token

        for match in re.finditer(token_pattern, result):
            token_name = match.group(1)
            
            token_value = getattr(self.tokenizer, token_name, "")
            result = result.replace("{{" + token_name + "}}", token_value)

        return result

    def _encode(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.template.default_system
        encoded_messages = []

        for i, message in enumerate(messages):
            current_message = ""
            
            if i == 0:
                current_message += self.template.format_prefix
                if system or tools:
                    tool_text = self._replace_tokens(self.template.format_tools, content=tools) if tools else ""
                    if self.template.system_style == "standard":
                        current_message += self._replace_tokens(self.template.format_system, content=(system + tool_text))

            if i > 0 and i % 2 == 0:
                current_message += self.template.format_separator

            if message["role"] == Role.USER:
                current_message += self._replace_tokens(
                    self.template.format_user,
                    content=message["content"] if self.template.system_style == "standard" else system + message["content"],
                    idx=str(i // 2)
                )
            elif message["role"] == Role.ASSISTANT:
                current_message += self._replace_tokens(
                    self.template.format_assistant,
                    content=message["content"]
                )
            elif message["role"] == Role.OBSERVATION:
                current_message += self._replace_tokens(
                    self.template.format_observation,
                    content=message["content"]
                )
            elif message["role"] == Role.FUNCTION:
                current_message += self._replace_tokens(
                    self.template.format_function,
                    content=message["content"]
                )
            else:
                raise NotImplementedError(f"Unexpected role: {message['role']}")

            encoded_messages.append(self.tokenizer.encode(current_message, add_special_tokens=False))
        return encoded_messages

    def encode_single_turn(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence[str],
        videos: Sequence[str],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ):
        r"""
        Returns a pair of token ids representing prompt and answer respectively.
        """
        messages = self.process_media_tokens(messages, images, videos)
        encoded_messages = self._encode(messages, system, tools)
        prompt_ids = []
        for encoded_message in encoded_messages[:-1]:
            prompt_ids += encoded_message
        
        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids


class Qwen2vlConverter(BaseConverter):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_sample_frames(self, total_frames, video_fps, **kwargs) -> int:
        # TODO: Implement this method
        pass

    @override
    def process_media_tokens(
        self,
        messages,
        images,
        videos,
        **kwargs,
    ):
        self._check_input(images, videos)
        image_processor = self.processor.image_processor
        merge_length: int = image_processor.merge_size ** 2
        visual_inputs = self.get_visual_inputs(images, videos, **kwargs)
        image_grid_thw = visual_inputs.get("image_grid_thw", [])
        video_grid_thw = visual_inputs.get("video_grid_thw", [])
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|vision_start|>{self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)}<|vision_end|>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)}<|vision_end|>",         
                    1,
                )
                num_video_tokens += 1

            message["content"] = content
        if len(images) != num_image_tokens:
            raise ValueError(f"Number of {IMAGE_PLACEHOLDER} tokens does not match the number of images.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"Number of {VIDEO_PLACEHOLDER} tokens does not match the number of videos.")
        return messages


class LlavaNextConverter(BaseConverter):
    @override
    def process_media_tokens(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> List[Dict[str, str]]:
        self._check_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        visual_inputs = self.get_visual_inputs(images, videos)

        if "image_sizes" in visual_inputs:
            image_sizes = iter(visual_inputs["image_sizes"])

        if "pixel_values" in visual_inputs:
            height, width = get_image_size(to_numpy_array(visual_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while self.image_token in content:
                image_size = next(image_sizes)
                orig_height, orig_width = image_size
                image_seqlen = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                if self.processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
                num_image_tokens += 1
                content = content.replace(self.image_token, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))
        return messages


class MllamaConverter(BaseConverter):
    @override
    def process_media_tokens(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> List[Dict[str, str]]:
        self._check_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_visual_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        batch_ids: Sequence[List[int]],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._check_input(images, videos)
        if len(images) != len(batch_ids):
            raise ValueError("Mllama only supports one image per sample.")

        image_processor = self.processor.image_processor
        images = self._get_images(images, image_resolution=getattr(self.processor, "image_resolution", 512 * 512))
        visual_inputs = image_processor([[image] for image in images], return_tensors="pt")
        num_tiles = visual_inputs.pop("num_tiles")
        image_token_id = getattr(self.processor, "image_token_id")
        max_image_tiles = getattr(self.processor.image_processor, "max_image_tiles")
        cross_attention_token_mask = [
            get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
        ]
        visual_inputs["cross_attention_mask"] = torch.from_numpy(
            convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=max_image_tiles,
                length=max(len(input_ids) for input_ids in batch_ids),
            )
        )  # shape: (batch_size, length, max_num_images, max_num_tiles)
        return visual_inputs


class PixtralConverter(BaseConverter):

    @override
    def process_media_tokens(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> List[Dict[str, str]]:
        self._check_input(images, videos)

        patch_size = getattr(self.processor, "patch_size")
        image_token = getattr(self.processor, "image_token")
        image_break_token = getattr(self.processor, "image_break_token")
        image_end_token = getattr(self.processor, "image_end_token")

        num_image_tokens = 0
        messages = deepcopy(messages)
        visual_inputs = super().get_visual_inputs(images, videos)
        image_input_sizes = visual_inputs.get("image_sizes", None)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if image_input_sizes is None:
                    raise ValueError("Cannot get image input sizes.")

                if self.expand_mm_tokens:
                    image_size = image_input_sizes[0][num_image_tokens]
                    height, width = image_size
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [[image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)
                num_image_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_visual_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        batch_ids: Sequence[List[int]],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._check_input(images, videos)

        visual_inputs = super().get_visual_inputs(images, videos)
        if visual_inputs.get("pixel_values"):
            visual_inputs["pixel_values"] = visual_inputs["pixel_values"][0]

        visual_inputs.pop("image_sizes", None)
        return visual_inputs


class InternVLChatConverter(BaseConverter):

    @override
    def process_media_tokens(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> List[Dict[str, str]]:
        """Process media tokens for InternVL model.
        
        This converter replaces IMAGE_PLACEHOLDER with:
        <img><IMG_CONTEXT>...<IMG_CONTEXT></img>
        """

        self._check_input(images, videos)
        num_image_tokens = 0  # count of image tokens
        messages = deepcopy(messages)

        num_image_token = getattr(self.processor, "num_image_token")  # number of image tokens per image

        # Calculate number of image tokens needed
        visual_inputs = self.get_visual_inputs(images, videos)
        num_patches = visual_inputs['pixel_values'].size(0)
        
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(images):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")
                
                # Replace placeholder with image tokens
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<img>{self.image_token * num_patches * num_image_token}</img>",
                    1,
                )
                num_image_tokens += 1
            
            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"Number of {IMAGE_PLACEHOLDER} tokens does not match the number of images.")
        # print('num_image_token:', num_image_token, 'num_patches:', num_patches)
        return messages


CONVERTERS = {
    "default": BaseConverter,
    "qwen2_vl": Qwen2vlConverter,
    "llava_next": LlavaNextConverter,
    "llama3.2_vision": MllamaConverter,
    "pixtral": PixtralConverter,
    "internvl2_5": InternVLChatConverter
}


def get_mm_converter(
    name: str,
    template,
    processor,
    tokenizer
):
    converter_class = CONVERTERS.get(name, None)
    if converter_class is None:
        raise ValueError(f"Invalid converter name: {name}")

    return converter_class(template, processor, tokenizer)


def load_converter(processor, tokenizer, data_args):
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    converter = get_mm_converter(data_args.template, template, processor, tokenizer)

    return converter