from typing import List, Optional, Union

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InternVLChatProcessor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a InternVL image processor and a tokenizer into a single processor.

    Args:
        image_processor (`InternVLImageProcessor`):
            The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): 
            A Jinja template which will be used to format chat conversations.
        image_token (`str`, *optional*, defaults to "<image>"):
            Special token used to denote image placeholders in text.
    """
    attributes = []
    image_processor_class = "InternVLImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, image_token="<image>", **kwargs):
        super().__init__()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_end_token: bool = True,
        padding: bool = False,
        **kwargs
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None and images is not None and self.image_token not in text:
            logger.warning(
                f"You have passed both `text`: {text} and `images`: {len(images)} but no `{self.image_token}` "
                "token found in the text. The image will be processed but not properly matched with text."
            )

        encoding = {}

        if images is not None:
            image_features = self.image_processor(images, **kwargs)
            encoding.update(image_features)

        if text is not None:
            text_features = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=padding,
                **kwargs
            )
            encoding.update(text_features)

        # TODO: convert image placeholder to image token

        return BatchFeature(data=encoding)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = getattr(self.tokenizer, "model_input_names", [])
        image_processor_input_names = getattr(self.image_processor, "model_input_names", [])
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))