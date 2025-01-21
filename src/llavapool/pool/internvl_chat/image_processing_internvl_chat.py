from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    is_valid_image,
    validate_preprocess_arguments,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406) 
IMAGENET_STD = (0.229, 0.224, 0.225)


# Copied from transformers.models.llava_next.image_processing_llava_next.make_batched_images
def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")


class InternVLChatImageProcessor(BaseImageProcessor):
    """
    Image processor for InternVL models. Handles dynamic tiling and preprocessing of images.
    
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified size.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of output image after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_MEAN`):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STD`):
            Standard deviation to use if normalizing the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_tiles (`int`, *optional*, defaults to 1):
            Minimum number of tiles to split image into.
        max_tiles (`int`, *optional*, defaults to 12):
            Maximum number of tiles to split image into.
        use_thumbnail (`bool`, *optional*, defaults to True):
            Whether to include a thumbnail of the full image in addition to tiles.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True, 
        rescale_factor: float = 1/255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_tiles: int = 1,
        max_tiles: int = 12,
        use_thumbnail: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 448, "width": 448}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.do_convert_rgb = do_convert_rgb
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail

    def find_closest_aspect_ratio(
        self, 
        aspect_ratio: float,
        width: int,
        height: int,
        image_size: int
    ) -> Tuple[int, int]:
        """
        Find closest supported aspect ratio for tiling.
        """
        target_ratios = set(
            (i, j) for n in range(self.min_tiles, self.max_tiles + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= self.max_tiles and i * j >= self.min_tiles
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def dynamic_tile(
        self,
        image: Image.Image,
        size: Dict[str, int]
    ) -> List[Image.Image]:
        """
        Splits image into dynamic tiles based on aspect ratio.
        """
        image_size = size["height"]  # Assuming square tiles
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        tiles = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            tile = resized_img.crop(box)
            tiles.append(tile)

        if self.use_thumbnail and len(tiles) > 1:
            thumbnail = image.resize((image_size, image_size))
            tiles.append(thumbnail)

        return tiles

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None, 
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_batched_images(images)
        
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # Convert to PIL Image for tiling
        images = [Image.fromarray(to_numpy_array(image)) if not isinstance(image, Image.Image) else image 
                 for image in images]

        # Dynamic tiling
        tiled_images = []
        for image in images:
            tiles = self.dynamic_tile(image, size)
            tiled_images.extend(tiles)

        # Convert back to numpy for standard processing
        tiled_images = [to_numpy_array(image) for image in tiled_images]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(tiled_images[0])

        # Convert images to numpy arrays with correct channel format
        tiled_images = [to_channel_dimension_format(image, ChannelDimension.FIRST) for image in tiled_images]

        if do_resize:
            tiled_images = [
                resize(
                    image=image,
                    size=(size["height"], size["width"]),
                    resample=resample.value if hasattr(resample, 'value') else resample,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in tiled_images
            ]

        if do_rescale:
            tiled_images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST
                )
                for image in tiled_images
            ]

        if do_normalize:
            tiled_images = [
                self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST
                )
                for image in tiled_images
            ]

        tiled_images = [
            to_channel_dimension_format(image, data_format, ChannelDimension.FIRST)
            if data_format != ChannelDimension.FIRST else image
            for image in tiled_images
        ]
        
        data = {"pixel_values": tiled_images}
        return BatchFeature(data=data, tensor_type=return_tensors)