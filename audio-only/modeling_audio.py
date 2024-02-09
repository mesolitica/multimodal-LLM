from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
import math
from transformers.activations import gelu
from typing import List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers import CONFIG_MAPPING
from transformers.modeling_outputs import BaseModelOutput
from transformers import GenerationConfig
from transformers import CLIPConfig, CLIPProcessor, CLIPModel, AutoModel
from transformers import WhisperConfig, WhisperPreTrainedModel, WhisperModel
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig


def most_frequent_element(tensor):
    flattened_list = tensor.flatten().tolist()
    counter = Counter(flattened_list)
    most_common_element = counter.most_common(1)[0][1]

    return most_common_element


class MM_LLMs_Config(PretrainedConfig):
    model_type = 'mm_llms'
    is_composition = True

    def __init__(
        self,
        audio_config=None,
        llm_config=None,
        **kwargs
    ):

        self.audio_config = audio_config
        self.llm_config = llm_config

        if isinstance(self.audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "whisper"
            )
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        if isinstance(self.llm_config, dict):
            llm_config["model_type"] = llm_config["model_type"] if "model_type" in llm_config else "llama"
            self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        super().__init__(**kwargs)


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, in_hidden_size, out_hidden_size, conv_kernel=None, conv_stride=3):
        super().__init__()

        self.conv_kernel = conv_kernel

        if conv_kernel:
            self.linear_1 = nn.Conv1d(
                in_hidden_size,
                out_hidden_size,
                kernel_size=conv_kernel,
                stride=conv_stride)
        else:
            self.linear_1 = nn.Linear(
                in_hidden_size,
                out_hidden_size,
                bias=True,
            )
        self.act = gelu
        self.linear_2 = nn.Linear(
            out_hidden_size,
            out_hidden_size,
            bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        if self.conv_kernel:
            hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MM_LLMs(PreTrainedModel):
    config_class = MM_LLMs_Config
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config, flash_attention=False, dtype=torch.float32):
        super().__init__(config)
        self.config = config

        self.audio_encoder = AutoModel.from_config(config.audio_config)

        self.llm = AutoModelForCausalLM.from_config(
            config.llm_config,
            use_flash_attention_2=flash_attention,
            torch_dtype=dtype,
        )

        self.audio_projector = LlavaMultiModalProjector(
            config.audio_config.d_model,
            config.llm_config.hidden_size,
            conv_kernel=40,
            conv_stride=3,
        )

    def forward(self,
                input_ids: torch.LongTensor = None,
                image_index: torch.LongTensor = None,
                audio_index: torch.LongTensor = None,
                image_starts: torch.int = None,
                image_ends: torch.int = None,
                audio_starts: torch.int = None,
                audio_ends: torch.int = None,
                images: torch.FloatTensor = None,
                audios: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                use_cache: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audios = audios.type(self.audio_encoder.dtype) if audios is not None else None

        model_inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids,
            image_index=image_index,
            audio_index=audio_index,
            image_starts=image_starts,
            image_ends=image_ends,
            audio_starts=audio_starts,
            audio_ends=audio_ends,
            images=images,
            audios=audios,
            attention_mask=attention_mask,
            labels=labels)

        print(input_ids.shape, model_inputs['inputs_embeds'].shape)
        outputs = self.llm(
            inputs_embeds=model_inputs['inputs_embeds'],
            attention_mask=model_inputs['attention_mask'],
            labels=model_inputs['labels'],
            return_dict=return_dict)

        return outputs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            images=None,
            audios=None,
            audio_starts=None,
            audio_ends=None,
            image_starts=None,
            image_ends=None,
            attention_mask=None,
            labels=None,
            audio_index=None,
            image_index=None,
            **kwargs):

        audio_features = self.encode_audio(
            audios) if audios is not None else None
        embed_tokens = self.llm.model.embed_tokens
        text_embeddings = embed_tokens(input_ids)
        batch_size = text_embeddings.shape[0]
        seq_len = text_embeddings.shape[1]
        embed_dim = text_embeddings.shape[2]

        if len(audio_index):
            max_count_audio = most_frequent_element(audio_index)
        else:
            max_count_audio = 0

        if audio_features is not None:
            seq_audio = audio_features.shape[1]
        else:
            seq_audio = 0

        audio_len = seq_audio * max_count_audio

        new_len = text_embeddings.shape[1] + audio_len
        final_embedding = torch.zeros(
            batch_size, new_len, embed_dim,
            device=text_embeddings.device,
            dtype=text_embeddings.dtype
        )
        final_embedding[:, :seq_len] = text_embeddings
        final_attention_mask = torch.zeros(
            batch_size, new_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        final_attention_mask[:, :seq_len] = attention_mask
        if labels is not None:
            final_labels = torch.full(
                (batch_size, new_len),
                -100,
                device=labels.device,
                dtype=labels.dtype
            )
            final_labels[:, :seq_len] = labels
        else:
            final_labels = None

        audio_id = int(audio_starts[0])

        where_is = torch.where(input_ids == audio_id)
        positions = defaultdict(int)
        b_audio = 0

        for i in range(len(where_is[0])):
            b, k = where_is[0][i], where_is[1][i]
            int_b = int(b)
            int_k = int(k)
            f = audio_features[b_audio]
            b_audio += 1

            c = torch.cat([final_embedding[b, :int_k + 1 + positions[int_b]],
                          f, text_embeddings[b, k + 1:]])
            final_embedding[b, :len(c)] = c
            final_attention_mask[b, :len(c)] = 1.0

            if labels is not None:
                ignore = torch.tensor([-100] * len(f), device=labels.device)
                c_label = torch.cat(
                    [final_labels[b, :int_k + 1 + positions[int_b]], ignore, labels[b, k + 1:]])
                final_labels[b, :len(c)] = c_label

            positions[int_b] += len(f)

        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": final_embedding,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": final_attention_mask,
            "labels": final_labels,
        }
        return model_inputs

    def encode_audio(self, audios):
        encoded = self.audio_encoder.encoder(audios)[0]
        audio_features = self.audio_projector(encoded.transpose(1, 2).contiguous())
        return audio_features
