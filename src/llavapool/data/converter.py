
import math
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Union, Sequence
from PIL import Image
import numpy as np
from decord import VideoReader
from typing_extensions import override

from ..utils.constants import IMAGE_TOKEN, VIDEO_TOKEN
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


def smart_resize(
    image,
    resized_height: int = None,
    resized_width: int = None,
    max_pixels: int = MAX_PIXELS,
    min_pixels: int = MIN_PIXELS,
    factor: int = IMAGE_FACTOR,
):
    if resized_height is not None and resized_width is not None:
        height, width = resized_height, resized_width
    else:
        height, width = image.size

    adjusted_height = max(factor, round_by_factor(height, factor))
    adjusted_width = max(factor, round_by_factor(width, factor))
    if adjusted_height * adjusted_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        adjusted_height = floor_by_factor(height / beta, factor)
        adjusted_width = floor_by_factor(width / beta, factor)
    elif adjusted_height * adjusted_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        adjusted_height = ceil_by_factor(height * beta, factor)
        adjusted_width = ceil_by_factor(width * beta, factor)
    return image.resize((adjusted_width, adjusted_height))
        

class BaseConverter:
    def __init__(self, template, processor, image_token=None, video_token=None):
        self.template = template
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.image_token = image_token
        self.video_token = video_token

    def _check_input(self, images, videos):
        if len(images) >= 1 and self.image_token is None:
            raise ValueError("Image token is required for images")
        if len(videos) >= 1 and self.video_token is None:
            raise ValueError("Video token is required for videos")

    def convert(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def __call__(self):
        return self.convert()

    def _get_images(self, images, **kwargs):
        preprocessed_images = []

        for image in images:
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            
            if pil_image is None:
                raise ValueError("Invalid image type")
            
            pil_image = pil_image.convert("RGB")
            pil_image = smart_resize(pil_image, **kwargs)
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
    
    def get_mm_inputs(self, images, videos, **kwargs):
        image_processor = self.processor.image_processor
        video_processor = getattr(self.processor, 'video_processor', None) or image_processor

        input_dict = {"images": None}

        if len(images) != 0:
            input_dict["images"] = self._get_images(images, **kwargs)
        
        if len(videos) != 0:
            input_dict["videos"] = self._get_videos(videos, **kwargs)
        
        mm_inputs = {}
        if input_dict.get("images") is not None:
            mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
        if input_dict.get("videos") is not None:
            mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))

        return mm_inputs

    def process_media_tokens(self, messages, images, videos):
        raise NotImplementedError("This method must be implemented in the subclass")

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
        
    # def _replace_tokens(self, template: str, **kwargs) -> str:
    #     """Replace template variables with actual values."""
    #     result = template
    #     for key, value in kwargs.items():
    #         result = result.replace("{{" + key + "}}", str(value))
    #     return result

    def _replace_tokens(self, template: str, **kwargs) -> str:
        """템플릿의 특수 토큰과 변수를 치환"""
        result = template
        
        # 특수 토큰 처리
        special_tokens = {
            "{{bos_token}}": self.tokenizer.bos_token or "",
            "{{eos_token}}": self.tokenizer.eos_token or "",
        }
        
        for token, value in special_tokens.items():
            result = result.replace(token, value)
        
        # 변수 치환
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
            
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
            
            # 첫 메시지 처리
            if i == 0:
                current_message += self.template.format_prefix  # 변경: .apply() 제거
                if system or tools:
                    tool_text = self._replace_tokens(self.template.format_tools, content=tools) if tools else ""
                    current_message += self._replace_tokens(self.template.format_system, content=(system + tool_text))

            # 구분자 추가
            if i > 0 and i % 2 == 0:
                current_message += self.template.format_separator  # 변경: .apply() 제거

            # 역할별 메시지 처리
            if message["role"] == Role.USER:
                current_message += self._replace_tokens(
                    self.template.format_user,
                    content=message["content"],
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

            # 토크나이징
            encoded_messages.append(self.tokenizer.encode(current_message, add_special_tokens=False))
            # import pdb;pdb.set_trace()
        return encoded_messages

    def _convert_elements_to_ids(self, elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.vkfd
        """
        token_ids = []

        
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += self.tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [self.tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and self.tokenizer.bos_token_id is not None:
                    token_ids += [self.tokenizer.bos_token_id]
                elif "eos_token" in elem and self.tokenizer.eos_token_id is not None:
                    token_ids += [self.tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids
    

class Qwen2vlConverter(BaseConverter):
    def __init__(self, template, processor, image_token=None, video_token=None):
        super().__init__(template, processor, "<|image_pad|>", "<|video_pad|>")

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
        mm_inputs = self.get_mm_inputs(images, videos, **kwargs)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_TOKEN in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_TOKEN} tokens.")

                content = content.replace(
                    IMAGE_TOKEN,
                    f"<|vision_start|>{self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)}<|vision_end|>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_TOKEN in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_TOKEN} tokens.")

                content = content.replace(
                    VIDEO_TOKEN,
                    "<|vision_start|>{self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)}<|vision_end|>",         
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"Number of {IMAGE_TOKEN} tokens does not match the number of images.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"Number of {VIDEO_TOKEN} tokens does not match the number of videos.")

        return messages


CONVERTERS = {
    "base": BaseConverter,
    "qwen2_vl": Qwen2vlConverter,
}


def get_mm_converter(
    name: str,
    template,
    processor,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
):
    converter_class = CONVERTERS.get(name, None)
    if converter_class is None:
        raise ValueError(f"Invalid converter name: {name}")

    return converter_class(template, processor, image_token, video_token)



def load_converter(processor, data_args, image_token=None, video_token=None):
    
    template = get_template_and_fix_tokenizer(processor.tokenizer, data_args)
    converter = get_mm_converter(data_args.template, template, processor)

    return converter