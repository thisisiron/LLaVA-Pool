"""Dataset module for vision-language model training."""

import copy
import os
from typing import Dict, List, Optional, Union

import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)
from qwen_vl_utils import process_vision_info

from ..config.params import DataArguments
from ..utils.constants import (
    MODEL_TYPES,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    EOT_TOKEN,
    PHI_EOT_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    SYSTEM_MESSAGE,
)


def encode_video(video_path: str, max_num_frames: int = 10) -> List[Image.Image]:
    """Encode video into frames.

    Args:
        video_path: Path to video file
        max_num_frames: Maximum number of frames to extract

    Returns:
        List of PIL Image frames
    """
    def uniform_sample(sequence: List, n: int) -> List:
        gap = len(sequence) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [sequence[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)

    frames = vr.get_batch(frame_idx).asnumpy()
    return [Image.fromarray(v.astype('uint8')) for v in frames]


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """Pad a list of sequences to the same length.

    Args:
        sequences: list of tensors in [seq_len, *] shape
        padding_side: which side to pad ('right' or 'left')
        padding_value: value to pad with

    Returns:
        Padded tensor
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full(
        (batch_size, max_len) + trailing_dims,
        padding_value
    )
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel):
    """Get image information using process_vision_info.

    Args:
        image_path: Path to image file
        min_pixel: Minimum pixel size
        max_pixel: Maximum pixel size

    Returns:
        Processed image information
    """
    messages = [{
        "role": "user",
        "content": [{
            "type": "image",
            "image": image_path,
            "min_pixel": min_pixel,
            "max_pixel": max_pixel
        }]
    }]

    image_input, _ = process_vision_info(messages)
    return image_input[0]


def get_video_info(video_path, max_pixels, fps):
    """Get video information using process_vision_info.

    Args:
        video_path: Path to video file
        max_pixels: Maximum pixel size
        fps: Frames per second

    Returns:
        Processed video information
    """
    messages = [{
        "role": "user",
        "content": [{
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
            "fps": fps
        }]
    }]

    _, video_input = process_vision_info(messages)
    return video_input[0]


class BaseVisionLanguageDataset(Dataset):
    """Base dataset class for vision-language models."""
    
    def __init__(
        self,
        data_path: Union[str, list],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_type: str,
        padding: bool = True,
    ):
        super().__init__()
        
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.list_data_dict = (json.load(open(data_path, "r")) 
                              if isinstance(data_path, str) else data_path)
        self.processor = processor
        self.data_args = data_args
        self.padding = padding
        self.model_type = model_type
        
        # Model specific settings
        if model_type == "qwen2-vl":
            self.min_pixel = data_args.min_pixels
            self.max_pixel = data_args.max_pixels
            self.fps = data_args.fps
        elif model_type in ["llama", "phi"]:
            self.max_num_frames = data_args.max_num_frames

    def __len__(self):
        return len(self.list_data_dict)

    def _process_media(self, sources: Dict) -> Dict:
        """Process image or video inputs based on model type."""
        is_video = False
        num_frames = None
        images = None
        videos = None
        media_info = {}

        if "image" in sources:
            image_files = sources["image"]
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(self.data_args.image_folder, image_file)
                
                if self.model_type == "qwen":
                    images.append(get_image_info(image_file, self.min_pixel, self.max_pixel))
                else:
                    images.append(Image.open(image_file).convert("RGB"))

            media_info["grid_key"] = "image_grid_thw" if self.model_type == "qwen2-vl" else None
            media_info["pixel_key"] = "pixel_values"

        elif "video" in sources:
            is_video = True
            video_file = sources["video"]
            if not os.path.exists(video_file):
                if not video_file.startswith("http"):
                    video_file = os.path.join(self.data_args.image_folder, video_file)

            if self.model_type == "qwen2-vl":
                videos = [get_video_info(video_file, self.max_pixel, self.fps)]
                media_info["grid_key"] = "video_grid_thw"
                media_info["pixel_key"] = "pixel_values_videos"
            else:
                images = encode_video(video_file, self.max_num_frames)
                num_frames = len(images)
                media_info["grid_key"] = None
                media_info["pixel_key"] = "pixel_values"
        
        return {
            "is_video": is_video,
            "num_frames": num_frames,
            "images": images,
            "videos": videos,
            **media_info
        }

    def _process_conversation(
        self,
        sources: List[Dict],
        media_info: Dict,
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process conversation based on model type."""
        if self.model_type == "qwen2-vl":
            return self._process_qwen_conversation(sources, media_info, idx)
        elif self.model_type == "llama":
            return self._process_llama_conversation(sources, media_info, idx)
        elif self.model_type == "phi":
            return self._process_phi_conversation(sources, media_info, idx)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        media_info = self._process_media(sources)
        
        sources = copy.deepcopy(
            llava_to_openai(
                sources['conversations'],
                is_video=media_info["is_video"],
                num_frames=media_info["num_frames"],
                model_type=self.model_type
            )
        )
        
        return self._process_conversation(sources, media_info, i)

    def _process_qwen_conversation(
        self,
        sources: List[Dict],
        media_info: Dict,
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process conversation for Qwen model."""
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []

        if SYSTEM_MESSAGE:
            system_message = (
                f"{DEFAULT_IM_START_TOKEN}system\n"
                f"{SYSTEM_MESSAGE}\n"
                f"{DEFAULT_IM_END_TOKEN}\n"
            )
            system_ids = self.processor.tokenizer(
                system_message,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids']
            system_labels = torch.full_like(system_ids, IGNORE_INDEX)
            
            all_input_ids.append(system_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for conv_idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            assistant_response = sources[j + 1]

            user_text = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
                f"{user_input['content']}\n"
                f"{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{assistant_response['role']}\n"
            )
            assistant_text = (
                f"{assistant_response['content']}\n"
                f"{DEFAULT_IM_END_TOKEN}\n"
            )

            if conv_idx == 0:
                inputs = self.processor(
                    text=[user_text],
                    images=media_info["images"],
                    videos=media_info["videos"],
                    padding=False,
                    return_tensors='pt'
                )
                prompt_ids = inputs['input_ids']
                all_pixel_values.append(inputs[media_info["pixel_key"]])
                all_image_grid_thw.append(inputs[media_info["grid_key"]])
            else:
                prompt_ids = self.processor.tokenizer(
                    user_text,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors='pt'
                )['input_ids']

            response_ids = self.processor.tokenizer(
                assistant_text,
                add_special_tokens=False,
                padding=False,
                return_tensors='pt'
            )['input_ids']

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat([
                torch.full((len(prompt_ids[0]),), IGNORE_INDEX),
                response_ids.squeeze(0)
            ], dim=0)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)
        attention_mask = torch.ones_like(input_ids)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        if media_info["pixel_key"] and media_info["grid_key"]:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[media_info["pixel_key"]] = pixel_values
            data_dict[media_info["grid_key"]] = image_thw

        return data_dict

    def _process_llama_conversation(
        self,
        sources: List[Dict],
        media_info: Dict,
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process conversation for Llama model."""
        all_input_ids = []
        all_labels = []
        pixel_values = None
        aspect_ratio_ids = None
        aspect_ratio_mask = None
        cross_attention_mask = None

        if media_info["images"] is not None:
            input_text = self.processor.apply_chat_template(
                sources,
                add_generation_prompt=False
            )
            inputs = self.processor(
                images=media_info["images"],
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            pixel_values = inputs['pixel_values']
            aspect_ratio_ids = inputs['aspect_ratio_ids']
            aspect_ratio_mask = inputs['aspect_ratio_mask']
            cross_attention_mask = inputs['cross_attention_mask'].squeeze(0)

        for conv_idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            assistant_response = sources[j + 1]
            response_text = f"{assistant_response['content'][0]['text']}{EOT_TOKEN}"

            if conv_idx == 0:
                user_prompt = self.processor.apply_chat_template(
                    [user_input],
                    add_generation_prompt=True
                )
                prompt_ids = self.processor.tokenizer(
                    user_prompt,
                    add_special_tokens=False,
                    return_tensors='pt'
                )['input_ids']
            else:
                user_prompt = (
                    f"{START_HEADER_TOKEN}{user_input['role']}"
                    f"{END_HEADER_TOKEN}\n\n"
                    f"{user_input['content'][0]['text']}{EOT_TOKEN}"
                    f"{START_HEADER_TOKEN}{assistant_response['role']}"
                    f"{END_HEADER_TOKEN}\n\n"
                )
                prompt_ids = self.processor.tokenizer(
                    user_prompt,
                    add_special_tokens=False,
                    return_tensors='pt'
                )['input_ids']

            response_ids = self.processor.tokenizer(
                response_text,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids']

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat([
                torch.full((len(prompt_ids[0]),), IGNORE_INDEX),
                response_ids.squeeze(0)
            ], dim=0)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)
        attention_mask = torch.ones_like(input_ids)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'aspect_ratio_ids': aspect_ratio_ids,
            'aspect_ratio_mask': aspect_ratio_mask,
            'cross_attention_mask': cross_attention_mask
        }

        return data_dict

    def _process_phi_conversation(
        self,
        sources: List[Dict],
        media_info: Dict,
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process conversation for Phi model."""
        bos_token = self.processor.tokenizer.bos_token_id
        all_input_ids = [torch.tensor([bos_token])]
        all_labels = [torch.tensor([IGNORE_INDEX])]
        all_pixel_values = []
        all_image_sizes = []

        for conv_idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            assistant_response = sources[j + 1]

            if isinstance(user_input['content'], list):
                user_content = user_input['content'][0].get('text', '')
            else:
                user_content = user_input['content']

            # if self.processor.tokenizer.apply_chat_template is None:
            #     user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{assistant_response['role']}\n"

            user_text = self.processor.tokenizer.apply_chat_template(
                [user_input],
                tokenize=False,
                add_generation_prompt=True
            )
            assistant_text = f"{assistant_response['content']}{PHI_EOT_TOKEN}"

            if conv_idx == 0:
                inputs = self.processor(
                    text=user_text,
                    images=media_info["images"],
                    return_tensors='pt'
                )
                prompt_ids = inputs['input_ids']
                all_pixel_values.append(inputs.get('pixel_values'))
                all_image_sizes.append(inputs.get('image_sizes'))
            else:
                prompt_ids = self.processor.tokenizer(
                    user_text,
                    add_special_tokens=False,
                    return_tensors='pt'
                )['input_ids']
            response_ids = self.processor.tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids']

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat([
                torch.full((len(prompt_ids[0]),), IGNORE_INDEX),
                response_ids.squeeze(0)
            ], dim=0)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Add EOS token using tokenizer
        eos_token_id = self.processor.tokenizer.eos_token_id
        assert eos_token_id == 32000, f"EOS token ID {eos_token_id} does not match expected value 32000"
        all_input_ids.append(torch.tensor([eos_token_id]))
        all_labels.append(torch.tensor([eos_token_id]))
        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)
        attention_mask = torch.ones_like(input_ids)

        pixel_values = (
            torch.cat([pv for pv in all_pixel_values if pv is not None and pv.numel() > 0], dim=0)
            if any(pv is not None and pv.numel() > 0 for pv in all_pixel_values)
            else None
        )
        image_sizes = (
            torch.cat([isize for isize in all_image_sizes if isize is not None and isize.numel() > 0], dim=0)
            if any(isize is not None and isize.numel() > 0 for isize in all_image_sizes)
            else None
        )

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_sizes': image_sizes
        }

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, processor: transformers.ProcessorMixin = None):
        self.pad_token_id = pad_token_id
        self.processor = processor

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 모든 모델에 공통적으로 필드들
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []

        # 모델별 특수 필드들
        batch_aspect_ratio_ids = []
        batch_aspect_ratio_mask = []
        batch_cross_attention_mask = []
        batch_image_sizes = []
        batch_image_grid_thw = []
        batch_video_grid_thw = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            
            # 이미지/비디오 관련 필드들
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
            if "pixel_values_videos" in example:
                batch_pixel_values.append(example["pixel_values_videos"])
            if "aspect_ratio_ids" in example:
                batch_aspect_ratio_ids.append(example["aspect_ratio_ids"])
            if "aspect_ratio_mask" in example:
                batch_aspect_ratio_mask.append(example["aspect_ratio_mask"])
            if "cross_attention_mask" in example:
                batch_cross_attention_mask.append(example["cross_attention_mask"])
            if "image_sizes" in example:
                batch_image_sizes.append(example["image_sizes"])
            if "image_grid_thw" in example:
                batch_image_grid_thw.append(example["image_grid_thw"])
            if "video_grid_thw" in example:
                batch_video_grid_thw.append(example["video_grid_thw"])

        # 기본 필드들 패딩
        input_ids = pad_sequence(
            batch_input_ids,
            padding_side='right',
            padding_value=self.pad_token_id
        )
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(
            batch_label_ids,
            padding_side='right',
            padding_value=IGNORE_INDEX
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 이미지/비디오 관련 필드들 처리
        if batch_pixel_values:
            pixel_values = torch.cat([pv for pv in batch_pixel_values if pv is not None and pv.numel() > 0], dim=0)
            if pixel_values.numel() > 0:
                batch["pixel_values"] = pixel_values

        if batch_aspect_ratio_ids:
            aspect_ratio_ids = torch.cat([ar for ar in batch_aspect_ratio_ids if ar is not None and ar.numel() > 0], dim=0)
            if aspect_ratio_ids.numel() > 0:
                batch["aspect_ratio_ids"] = aspect_ratio_ids

        if batch_aspect_ratio_mask:
            aspect_ratio_mask = torch.cat([am for am in batch_aspect_ratio_mask if am is not None and am.numel() > 0], dim=0)
            if aspect_ratio_mask.numel() > 0:
                batch["aspect_ratio_mask"] = aspect_ratio_mask

        if batch_cross_attention_mask:
            cross_attention_mask = pad_sequence(
                [cam for cam in batch_cross_attention_mask if cam is not None],
                padding_side='right',
                padding_value=0
            )
            batch["cross_attention_mask"] = cross_attention_mask

        if batch_image_sizes:
            image_sizes = torch.cat([isize for isize in batch_image_sizes if isize is not None and isize.numel() > 0], dim=0)
            if image_sizes.numel() > 0:
                batch["image_sizes"] = image_sizes

        if batch_image_grid_thw:
            image_grid = torch.cat([ig for ig in batch_image_grid_thw if ig is not None and ig.numel() > 0], dim=0)
            if image_grid.numel() > 0:
                batch["image_grid_thw"] = image_grid

        if batch_video_grid_thw:
            video_grid = torch.cat([vg for vg in batch_video_grid_thw if vg is not None and vg.numel() > 0], dim=0)
            if video_grid.numel() > 0:
                batch["video_grid_thw"] = video_grid

        return batch

def replace_image_tokens(input_string: str, model_type: str, start_count: int = 1) -> tuple:
    """
    Replace image tokens based on model type.
    
    Args:
        input_string: Input text containing image tokens
        model_type: Type of the model
        start_count: Starting count for image tokens
        
    Returns:
        Tuple of (processed string, image count, has image flag)
    """
    count = start_count
    has_image = False

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count, has_image

    while LLAVA_IMAGE_TOKEN in input_string:
        has_image = True
        if model_type == "qwen2-vl":
            replacement = f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}"
        elif model_type == "phi":
            replacement = f"<|image_{count}|>"
        else:  # LLAMA
            replacement = ""
        
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN, replacement, 1)
        count += 1

    return input_string, count, has_image

def video_to_image_tokens(input_string: str, num_frames: int, model_type: str) -> str:
    """
    Convert video token to image tokens based on model type.
    
    Args:
        input_string: Input text containing video token
        num_frames: Number of frames to extract
        model_type: Type of the model
        
    Returns:
        Processed string with video token replaced by image tokens
    """
    if model_type == "qwen2-vl":
        replacement = f"{VISION_START_TOKEN}{DEFAULT_VIDEO_TOKEN}{VISION_END_TOKEN}"
        return input_string.replace(LLAVA_VIDEO_TOKEN, replacement)
    else:
        frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
        return input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)


def llava_to_openai(
    conversations: List[Dict],
    is_video: bool = False,
    num_frames: Optional[int] = None,
    model_type: str = "llama"
) -> List[Dict]:
    """
    Convert LLaVA format conversations to OpenAI format.
    
    Args:
        conversations: List of conversation messages
        is_video: Flag indicating if input contains video
        num_frames: Number of video frames
        model_type: Type of the model
        
    Returns:
        Transformed conversation data
    """
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    image_count = 0 if model_type != "phi" else 1

    for conversation in conversations:
        if is_video:
            conversation['value'] = video_to_image_tokens(
                conversation["value"],
                num_frames,
                model_type
            )
        
        transformed_content, image_count, has_image = replace_image_tokens(
            conversation["value"],
            model_type,
            image_count
        )

        if model_type == "llama":
            content = []
            if has_image:
                for _ in range(image_count):
                    content.append({"type": "image"})
            content.append({"type": "text", "text": transformed_content})
        else:
            content = transformed_content

        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def make_supervised_data_module(
    processor: transformers.ProcessorMixin,
    data_args: DataArguments,
    model_type: str
) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    
    Args:
        processor: Model processor
        data_args: Data arguments
        model_type: Type of the model
        
    Returns:
        Dictionary containing dataset and collator
    """
    dataset = BaseVisionLanguageDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_type=model_type
    )
    
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        processor=processor
    )

    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": data_collator
    }