from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from transformers.utils import logging
from transformers.utils.versions import require_version


logger = logging.get_logger(__name__)


@dataclass
class Template:
    format_user: str
    format_assistant: str
    format_system: str
    format_tools: str
    format_separator: str
    format_prefix: str
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    system_style: str
    image_token: str
    video_token: str

# def _replace_tokens(tokenizer, format_str: str, **kwargs) -> str:
#     result = format_str

#     for key, value in kwargs.items():
#         result = result.replace("{{" + key + "}}", str(value))

#     token_pattern = r'\{\{(\w+_token)\}\}'  # ex. bos_token, eos_token

#     for match in re.finditer(token_pattern, result):
#         token_name = match.group(1)
        
#         token_value = getattr(tokenizer, token_name, None)
#         if token_value is None:
#             raise ValueError(f"Token '{token_name}' not found in tokenizer.")
#         result = result.replace("{{" + token_name + "}}", token_value)

#     return result


TEMPLATE: Dict[str, "Template"] = {}


def _register_template(
    name: str,
    format_user: Optional[str],
    format_assistant: Optional[str],
    format_system: Optional[str] = None,
    format_tools: Optional[str] = None,
    format_separator: Optional[str] = None,
    format_prefix: Optional[str] = None,
    default_system: str = "",
    image_token: str = "<image>",
    video_token: str = "<video>",
    stop_words: Sequence[str] = [],
    efficient_eos: bool = False,
    replace_eos: bool = False,  # replace eos_token with stop_words[0]
    replace_jinja_template: bool = True,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    # eos_token = "" if efficient_eos else "{{eos_token}}"
    # default_assistant_formatter = "{{content}}" + f"{eos_token}"
    default_separator_formatter = ""
    default_prefix_formatter = ""
    default_system_style = "standard"
    TEMPLATE[name] = Template(
        format_prefix=format_prefix or default_prefix_formatter,
        format_system=format_system,
        default_system=default_system,
        system_style=default_system_style,
        format_user=format_user,
        format_assistant=format_assistant,
        format_tools="",
        format_separator=format_separator or default_separator_formatter,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        image_token=image_token,
        video_token=video_token,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _jinja_escape(content: str) -> str:
    return content.replace("'", r"\'")


def _convert_str_to_jinja(template: str, tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:

    bos_value = tokenizer.bos_token if tokenizer.bos_token is not None else ""
    eos_value = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    
    # content를 기준으로 먼저 분할
    parts = template.split("{{content}}", maxsplit=1)
    if len(parts) == 1:
        # content가 없는 경우
        text = parts[0]
        text = text.replace("{{bos_token}}", bos_value)
        text = text.replace("{{eos_token}}", eos_value)
        return "'" + _jinja_escape(text) + "'" if text else ""
    
    # content가 있는 경우
    left_part, right_part = parts
    slot_items = []
    
    # left_part 처리
    if left_part:
        left_part = left_part.replace("{{bos_token}}", bos_value)
        left_part = left_part.replace("{{eos_token}}", eos_value)
        slot_items.append("'" + _jinja_escape(left_part) + "'")
    
    # content 추가
    slot_items.append(placeholder)
    
    # right_part 처리
    if right_part:
        right_part = right_part.replace("{{bos_token}}", bos_value)
        right_part = right_part.replace("{{eos_token}}", eos_value)
        slot_items.append("'" + _jinja_escape(right_part) + "'")
    
    return " + ".join(slot_items)


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    jinja_template = ""
    if template.format_prefix:
        prefix = _convert_str_to_jinja(template.format_prefix, tokenizer)
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    )

    # system message 처리 수정
    system_message = _convert_str_to_jinja(template.format_system, tokenizer, placeholder="system_message")
    # if not isinstance(template, Llama2Template):
    #     jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    # 메시지 루프 처리
    jinja_template += "{% for message in loop_messages %}"
    jinja_template += "{% set content = message['content'] %}"
    
    # Llama2Template 특별 처리
    # if isinstance(template, Llama2Template):
    #     jinja_template += "{% if loop.index0 == 0 and system_message is defined %}"
    #     jinja_template += "{% set content = " + system_message + " + message['content'] %}"
    #     jinja_template += "{% endif %}"

    # 사용자/어시스턴트 메시지 처리
    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_str_to_jinja(template.format_user, tokenizer)
    jinja_template += "{{ " + user_message + " }}"

    jinja_template += "{% elif message['role'] == 'assistant' %}"
    # separator를 문자열 연결로 처리
    assistant_template = template.format_assistant + template.format_separator
    assistant_message = _convert_str_to_jinja(assistant_template, tokenizer)
    jinja_template += "{{ " + assistant_message + " }}"
    
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    """
    if data_args.template in ["llama", "phiv3", "qwen2_vl"]:
        require_version("transformers>=4.45.0", "To fix: pip install transformers>=4.45.0")
        require_version("accelerate>=0.34.0", "To fix: pip install accelerate>=0.34.0")

    if data_args.template is None:
        raise ValueError("Template is not specified.")
    else:
        template = TEMPLATE.get(data_args.template, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(data_args.template))

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template template does not support `train_on_prompt`.")
    
    # Set image and video tokens
    if not hasattr(tokenizer, "image_token") and template.image_token:
        setattr(tokenizer, "image_token", template.image_token)
        setattr(tokenizer, "image_token_id", tokenizer.convert_tokens_to_ids(template.image_token))
    
    if not hasattr(tokenizer, "video_token") and template.video_token:
        setattr(tokenizer, "video_token", template.video_token)
        setattr(tokenizer, "video_token_id", tokenizer.convert_tokens_to_ids(template.video_token))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        import pdb;pdb.set_trace()
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    if tokenizer.chat_template is None or template.replace_jinja_template:
        try:
            tokenizer.chat_template = _get_jinja_template(template, tokenizer)
        except ValueError:
            logger.info("Cannot add this chat template to tokenizer.")

    return template


_register_template(
    name="default",
    format_system="{{content}}\n",
    format_user="Human: {{content}}\nAssistant:",
    format_assistant="{{content}}",
    format_separator="\n",
    image_token="<image>",
)

_register_template(
    name="llava_onevision",
    format_system="<|im_start|>system\n{{content}}<|im_end|>\n",
    format_user="<|im_start|>user {{content}}<|im_end|><|im_start|>assistant\n",
    format_assistant="{{content}}<|im_end|>",
    format_separator="\n",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
    image_token="<image>",
)


_register_template(
    name="llava_next",
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    format_system="{{content}}",
    format_user="USER: {{content}} ASSISTANT:",
    format_assistant="{{content}}{{eos_token}}",
    image_token="<image>",
)


_register_template(
    name="llama3.2_vision",
    format_prefix="<|begin_of_text|>",
    format_system="<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>",
    format_user=(
            "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    format_assistant="{{content}}<|eot_id|>",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
    replace_jinja_template=False,
    image_token="<|image|>",
)


_register_template(
    name="pixtral",
    format_prefix="<s>",
    format_user="[INST] {{content}} [/INST]",
    format_assistant="{{content}}",
    image_token="[IMG]",
    replace_jinja_template=False,
)


_register_template(
    name="qwen2_vl",
    default_system="You are a helpful assistant.",
    format_system="<|im_start|>system\n{{content}}<|im_end|>\n",
    format_user="<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
    format_assistant="{{content}}<|im_end|>",
    format_separator="\n",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    replace_jinja_template=False,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
)


_register_template(
    name="internvl2_5",
    default_system="You are a helpful assistant.",
    format_system="<s><|im_start|>system\n{{content}}<|im_end|>\n",
    format_user="<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
    format_assistant="{{content}}<|im_end|>",
    format_separator="\n",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<IMG_CONTEXT>"
)


# _register_template(
#     name="deepseekv2",
#     default_system=(
#         "You are a helpful assistant. "
#         "Please answer truthfully and write out your thinking step by step to be sure you get the right answer."
#     )
#     format_system="<s><|im_start|>system\n{{content}}<|im_end|>\n",
#     format_user="<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n",