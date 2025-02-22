from collections import defaultdict
from typing import List, Dict, Optional, Any

from .base import StrategyMixin, infer_seqlen
from transformers import logging, PreTrainedTokenizer, ProcessorMixin
from ...utils.constants import IGNORE_INDEX

logger = logging.get_logger(__name__)


class SupervisedStrategy(StrategyMixin):
    def _encode(
        self,
        prompt: List[Dict[str, str]],  # [{"role": "user", "content": "text"}, ...]
        response: List[Dict[str, str]],  # [{"role": "assistant", "content": "text"}, ...]
        images: Optional[str] = None,
        videos: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[str] = None,
        converter: str = None,
        tokenizer: "PreTrainedTokenizer" = None,
        processor: "ProcessorMixin" = None,
        cutoff_len: int = 1024,
        train_on_prompt: bool = False,
        mask_history: bool = False,
        data_args: "DataArguments" = None,
    ):
        input_ids, labels = [], []
        encoded_pairs = converter.encode_multi_turn(
            prompt + response, 
            images, 
            videos,
            system=system,
            tools=tools,
        )
        total_length = 1 if converter.template.efficient_eos else 0

        if mask_history:
            encoded_pairs = encoded_pairs[::-1]

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                source_len=len(source_ids),
                target_len=len(target_ids),
                cutoff_len=cutoff_len - total_length,
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if train_on_prompt:
                source_label = source_ids
            elif converter.template.efficient_eos:
                source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids
            
            if mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
        
        if converter.template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]
        
        return input_ids, labels

    def preprocess(
        self,
        examples: Dict[str, List[Any]],
        converter,
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_args: "DataArguments",
    ) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
                continue

            input_ids, labels = self._encode(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                converter=converter,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
                train_on_prompt=data_args.train_on_prompt,
                mask_history=data_args.mask_history,
                data_args=data_args,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])

        return model_inputs
