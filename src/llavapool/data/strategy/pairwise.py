from collections import defaultdict
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin, logging

from ...utils.constants import IGNORE_INDEX
from .base import StrategyMixin, infer_seqlen


logger = logging.get_logger(__name__)


class PairwiseStrategy(StrategyMixin):
    def _encode(
        self,
        prompt: List[Dict[str, str]],  # [{"role": "user", "content": "text"}, ...]
        response: List[Dict[str, str]],  # [{"role": "assistant", "content": "text"}, ...]
        images: Optional[str] = None,
        videos: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[str] = None,
        converter = None,
        tokenizer: "PreTrainedTokenizer" = None,
        processor: "ProcessorMixin" = None,
        cutoff_len: int = 1024,
        train_on_prompt: bool = False,
        mask_history: bool = False,
        data_args: "DataArguments" = None,
    ):
        # import pdb; pdb.set_trace()
        prompt_ids, chosen_ids = converter.encode_single_turn(
            prompt + [response[0]], 
            images, 
            videos,
            system=system,
            tools=tools,
        )
        _, rejected_ids = converter.encode_single_turn(
            prompt + [response[1]], 
            images, 
            videos,
            system=system,
            tools=tools,
        )

        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), data_args.cutoff_len
        )

        if converter.template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

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
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode(
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
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])

        return model_inputs