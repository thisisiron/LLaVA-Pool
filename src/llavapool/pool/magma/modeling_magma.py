from typing import Optional

import torch
from torch import nn
import torch.utils.checkpoint
from tqdm import tqdm
from transformers.models.auto import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from .configuration_magma import MagmaConfig
from .visual_encoders import build_encoder
from .utils import check_local_file
from .utils import unwrap_peft
from ...utils.logging import get_logger

from .projectors import (
    CAbstractor,
    DAbstractor,
    MLPProjector,
)

logger = get_logger()


def apply_delta(base_model, delta_model_name_or_path):
    # Reference: fastchat/model/apply_delta.py from https://github.com/lm-sys/FastChat (vicuna)
    print(f"Loading the delta weights from {delta_model_name_or_path}")
    local_files_only, delta_file_name = check_local_file(delta_model_name_or_path)
    delta, loading_info = AutoModelForCausalLM.from_pretrained(
        delta_file_name,
        local_files_only=local_files_only,
        output_loading_info=True,
    )
    print("[Loading info for delta model] \n", loading_info)
    print("Applying the delta ...")
    for name, param in tqdm(base_model.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    return base_model


def get_media_indices(my_list):
    """Find media token (image, video, ...) starting indices.
    media token is negative number: -1, -2, ...
    """
    if isinstance(my_list, torch.Tensor):
        my_list = my_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
            result.append(i)
    return result


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class MagmaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = MagmaConfig
    base_model_prefix = "magma"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = [
        "CLIPEncoderLayer",
        "LlamaDecoderLayer",
        "MagmaVisualProjectorLayer",
        "LlamaForCausalLM",
        "Parameter",
    ]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if (
            isinstance(module, nn.Conv2d)  # noqa: SIM101
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            raise ValueError()


def build_projector(config: MagmaConfig):
        """Build projector (abstractor) and query_tokens (optionally for resampler)"""
        # proj_config = config.projector_config
        proj_type = config.projector_type
        num_input_tokens = config.vision_config.num_input_tokens

        abstractor = {
            "mlp": MLPProjector,
            "c-abs": CAbstractor,
            "d-abs": DAbstractor,
        }[proj_type](config, num_input_tokens=num_input_tokens)

        # deformable attention only supports fp32
        # if proj_type == "d-abs":
        #     abstractor.to(torch.float)
        # else:
        #     abstractor.to(dtype)

        return abstractor
      

class MagmaForConditionalGeneration(MagmaPreTrainedModel):
    config_class = MagmaConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MagmaConfig, vision_model, language_model):
        super().__init__(config)
        self.vision_model = vision_model

        # prevent re-init by HF trainer; is this a nice way?
        def _set_hf_initialized(module):
            module._is_hf_initialized = True
        self.vision_model.apply(_set_hf_initialized)

        # logger.info("Build projector ...")
        self.proj_type = config.projector_type
        self.abstractor = build_projector(config)

        # logger.info("Build LM ...")
        # self.language_model = self.build_language_model(config, init_kwargs)
        self.language_model = language_model
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        for module in [self.vision_model, self.abstractor, self.language_model]:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        for module in [self.vision_model, self.abstractor, self.language_model]:
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    def _get_input_dtype(self):
        dtype = unwrap_peft(self.vision_model).get_dtype()

        return dtype

    def _forward_and_project_vision_for_analysis(self, pixel_values):
        """Forward pixel_values & project (abstract) the visual features to LLM embedding space."""
        assert pixel_values is not None

        # =================================================== #
        # Forward vision model
        # =================================================== #
        visual_features = self.forward_vision(pixel_values)

        # =================================================== #
        # Forward projector
        # =================================================== #
        if self.proj_type == "resampler":
            visual_embeds = self.abstractor(
                encoder_hidden_states=visual_features,
                output_attentions=True,
            )
            info_anal = visual_embeds["attentions"]
            visual_embeds = visual_embeds["last_hidden_state"]
        elif self.proj_type == "d-abs":
            visual_embeds = self.abstractor(
                visual_features, output_attentions=True, output_sampling_locations=True
            )

            sampling_locations = visual_embeds["sampling_locations"]
            cross_attentions = visual_embeds["cross_attentions"]
            info_anal = (sampling_locations, cross_attentions)

            visual_embeds = visual_embeds["last_hidden_state"]
        else:
            raise NotImplementedError()

        # visual_embeds: [B, L, dim]
        return visual_embeds, info_anal

    def _get_visual_feature_at(self, v_output, layer_index):
        if type(layer_index) == list:  # multi-scale feature case
            visual_features = torch.stack(v_output, dim=1)[:, layer_index]  # [B, n_scales, L, dim]
        else:
            visual_features = v_output[layer_index]  # [B, L, dim]
        return visual_features
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
    
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward_vision(self, pixel_values):
        v_outputs = self.vision_model(pixel_values, return_dict=True, output_hidden_states=True)
        layer_index = self.config.projector_config.feature_layer_index
        visual_features = self._get_visual_feature_at(v_outputs.hidden_states, layer_index)
        visual_features = self.vision_model.postprocess_for_projector(visual_features)
        return visual_features

    def forward_projector(self, visual_features):
        visual_embeds = self.abstractor(visual_features)["last_hidden_state"]

        return visual_embeds

    def forward_and_project_vision(self, pixel_values):
        """Forward pixel_values & project (abstract) the visual features to LLM embedding space."""
        assert pixel_values is not None
        visual_features = self.forward_vision(pixel_values)
        visual_embeds = self.forward_projector(visual_features)

        # visual_embeds: [B, L, dim]
        return visual_embeds

    def embed_text_tokens(self, input_ids, inplace=False):
        """Embed input_ids into text_embeds, ignoring media tokens (negative values).
        """
        if not inplace:
            input_ids = input_ids.clone()
        input_ids[input_ids < 0] = 0

        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        if hasattr(self.language_model, "transformer") and hasattr(
            self.language_model.transformer, "word_embeddings_layernorm"
        ):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        return text_embeds

    def prepare_mm_inputs(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """Prepare multimodal inputs from input_ids and pixel_values.
        """
        if pixel_values is not None:
            pixel_values = pixel_values.to(self._get_input_dtype())

        if attention_mask is None:
            attention_mask = input_ids.new_ones(*input_ids.shape)

        # Get Text Embeddings
        text_embeds = self.embed_text_tokens(input_ids)

        # Get Visual Embeddings
        if pixel_values is not None:
            visual_embeds = self.forward_and_project_vision(pixel_values)  # [B, L, lm_dim]
            img_seq_length = visual_embeds.shape[1]  # visual token length for single image
        else:
            img_seq_length = 0

        # get media token starting indices
        media_token_indices = [get_media_indices(ids) for ids in input_ids]
        num_images_per_sample = [len(x) for x in media_token_indices]

        # sanity check (assume all media tokens are image tokens)
        n_vision_tokens = (input_ids == -1).sum(1)  # -1 is image token
        num_images = torch.as_tensor(num_images_per_sample, device=n_vision_tokens.device)
        assert (
            (n_vision_tokens == num_images * img_seq_length).all().item()
        ), f"Expected #img_tokens={n_vision_tokens}, but got {num_images * img_seq_length}"

        # Get Actual Multimodal Input Embeddings
        batch_size = input_ids.shape[0]
        input_chunk_embeds = []
        input_chunk_attns = []
        img_idx = 0
        for b in range(batch_size):
            start = 0
            embeds = []
            attns = []

            for i, pos in enumerate(media_token_indices[b]):
                if pos > start:
                    embeds.append(text_embeds[b, start:pos])  # add tokens before visual tokens
                    attns.append(attention_mask[b, start:pos])
                embeds.append(visual_embeds[img_idx + i])  # add visual tokens
                img_embed_attn_mask = torch.ones(
                    visual_embeds[img_idx + i].shape[0], device=visual_embeds.device
                )
                attns.append(img_embed_attn_mask)
                start = pos + img_seq_length

            if start < text_embeds.shape[1]:
                embeds.append(text_embeds[b, start:])  # add instruction & response
                attns.append(attention_mask[b, start:])

            img_idx += num_images_per_sample[b]
            input_chunk_embeds.append(torch.cat(embeds, dim=0))
            input_chunk_attns.append(torch.cat(attns, dim=0))

        input_embeds = torch.stack(input_chunk_embeds, dim=0)
        attention_mask = torch.stack(input_chunk_attns, dim=0)

        return {
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        # num_images: torch.LongTensor,
        seq_length: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs = self.prepare_mm_inputs(
            input_ids,
            pixel_values,
            attention_mask,
        )
        input_embeds = inputs["input_embeds"]

        # Forward into LM
        # loss is computed in forwarding LM
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            output_attentions=self.config.output_attentions,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        seq_length: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if input_ids is None:
            return self.language_model.generate(attention_mask=attention_mask, **generate_kwargs)

        inputs = self.prepare_mm_inputs(
            input_ids,
            pixel_values,
            attention_mask,
        )
        inputs_embeds = inputs["input_embeds"]
        attention_mask = inputs["attention_mask"]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            input_shape = input_ids.shape
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "is_decoder": True,
        }
