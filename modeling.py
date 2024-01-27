from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
import math
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

    def __init__(self, attention_heads=8, image_conv_kernel=48, image_conv_stride=36,
                 audio_conv_kernel=240, audio_conv_stride=220,
                 image_config=None, audio_config=None, llm_config=None, **kwargs):

        self.image_config = image_config
        self.audio_config = audio_config
        self.llm_config = llm_config
        self.attention_heads = attention_heads
        self.image_conv_kernel = image_conv_kernel
        self.image_conv_stride = image_conv_stride
        self.audio_conv_kernel = audio_conv_kernel
        self.audio_conv_stride = audio_conv_stride

        if isinstance(self.image_config, dict):
            image_config["model_type"] = (
                image_config["model_type"] if "model_type" in image_config else "clip"
            )
            self.image_config = CONFIG_MAPPING[image_config["model_type"]](**image_config)
        if isinstance(self.audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "whisper"
            )
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        if isinstance(self.llm_config, dict):
            llm_config["model_type"] = llm_config["model_type"] if "model_type" in llm_config else "llama"
            self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.hidden_size = max(
            self.llm_config.hidden_size,
            self.image_config.vision_config.hidden_size,
            self.audio_config.d_model,
        )

        super().__init__(**kwargs)


class MM_LLMs(PreTrainedModel):
    config_class = MM_LLMs_Config
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.image_encoder = AutoModel.from_config(config.image_config)

        self.audio_encoder = AutoModel.from_config(config.audio_config)

        self.llm = AutoModelForCausalLM.from_config(config.llm_config)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        self.temporal_self_attention = nn.MultiheadAttention(
            config.image_config.text_config.hidden_size,
            config.attention_heads,
            dropout=attn_dropout,
            add_bias_kv=is_add_bias_kv,
            add_zero_attn=is_add_zero_attn)

        self.audio_align_attention = nn.MultiheadAttention(config.llm_config.hidden_size,
                                                           config.attention_heads * 2,
                                                           dropout=attn_dropout,
                                                           add_bias_kv=is_add_bias_kv,
                                                           add_zero_attn=is_add_zero_attn)

        self.image_align_attention = nn.MultiheadAttention(config.llm_config.hidden_size,
                                                           config.attention_heads * 2,
                                                           dropout=attn_dropout,
                                                           add_bias_kv=is_add_bias_kv,
                                                           add_zero_attn=is_add_zero_attn)

        self.transform_audio_to_hidden = nn.Linear(config.audio_config.d_model,
                                                   config.llm_config.hidden_size)
        self.transform_image_to_hidden = nn.Linear(config.image_config.text_config.hidden_size,
                                                   config.llm_config.hidden_size)

        self.project_image = nn.Conv1d(
            config.image_config.text_config.hidden_size,
            config.image_config.text_config.hidden_size,
            kernel_size=config.image_conv_kernel,
            stride=config.image_conv_stride)
        self.project_audio = nn.Conv1d(
            config.audio_config.d_model,
            config.audio_config.d_model,
            kernel_size=config.audio_conv_kernel,
            stride=config.audio_conv_stride)

        self.visual_projection = nn.Linear(
            self.image_encoder.vision_model.config.hidden_size,
            self.config.image_config.text_config.hidden_size,
            bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.layer_norm = nn.LayerNorm(config.image_config.text_config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)

        self.sigmoid = nn.Sigmoid()

        self.loss_fct = CrossEntropyLoss()

        self.init_weights()

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
                return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        images = images.type(self.image_encoder.dtype) if images is not None else None
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

        image_features = self.encode_image(
            images) if images is not None else None
        audio_features = self.encode_audio(
            audios) if audios is not None else None
        embed_tokens = self.llm.model.embed_tokens
        text_embeddings = embed_tokens(input_ids)

        token_embeddings = embed_tokens.weight.unsqueeze(0).repeat(
            text_embeddings.size(0), 1, 1).transpose(0, 1)

        ingore_num = 0

        if audio_features is not None:

            audio_starts = embed_tokens(audio_starts).unsqueeze(1)
            audio_ends = embed_tokens(audio_ends).unsqueeze(1)

            audio_features = self.project_audio(
                audio_features.transpose(
                    1, 2).contiguous()).transpose(
                1, 2).contiguous()

            audio_features = self.transform_audio_to_hidden(audio_features)

            max_count = most_frequent_element(audio_index)

            seq_img = audio_features.shape[1]
            dim = token_embeddings.shape[2]

            new_audio = torch.zeros(
                (token_embeddings.shape[1],
                 seq_img * max_count,
                 dim),
                device=token_embeddings.device,
                dtype=token_embeddings.dtype)
            current_dim = 0
            for no, index in enumerate(audio_index):
                if no > 0 and audio_index[no - 1] == index:
                    current_dim += 1
                else:
                    current_dim = 0
                new_audio[index, current_dim *
                          seq_img: (current_dim + 1) * seq_img] = audio_features[no]
                last_index = audio_index[0]

            audio_features = self.audio_align_attention(
                new_audio.transpose(
                    0,
                    1),
                token_embeddings,
                token_embeddings)[0].transpose(
                0,
                1).contiguous()

            audio_inputs = torch.cat(
                [torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)

            text_embeddings = torch.cat(
                [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audio_inputs], dim=1), text_embeddings[:, 1:, :]],
                dim=1)

            ingore_num += (audio_inputs.size(1))

        if image_features is not None:

            image_starts = embed_tokens(image_starts).unsqueeze(1)
            image_ends = embed_tokens(image_ends).unsqueeze(1)

            image_features = self.project_image(
                image_features.transpose(
                    1, 2).contiguous()).transpose(
                1, 2).contiguous()

            image_features = self.transform_image_to_hidden(image_features)

            max_count = most_frequent_element(image_index)

            seq_img = image_features.shape[1]
            dim = token_embeddings.shape[2]

            new_img = torch.zeros(
                (token_embeddings.shape[1],
                 seq_img * max_count,
                 dim),
                device=token_embeddings.device,
                dtype=token_embeddings.dtype)

            current_dim = 0
            for no, index in enumerate(image_index):
                if no > 0 and image_index[no - 1] == index:
                    current_dim += 1
                else:
                    current_dim = 0
                new_img[index, current_dim *
                        seq_img: (current_dim + 1) * seq_img] = image_features[no]
                last_index = image_index[0]

            image_features = self.image_align_attention(
                new_img.transpose(
                    0,
                    1),
                token_embeddings,
                token_embeddings)[0].transpose(
                0,
                1).contiguous()

            image_inputs = torch.cat(
                [torch.cat([image_starts, image_features], dim=1), image_ends], dim=1)

            text_embeddings = torch.cat(
                [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), image_inputs], dim=1),
                 text_embeddings[:, 1:, :]], dim=1)

            ingore_num += (image_inputs.size(1))

        if attention_mask is not None:
            attentionmask = torch.tensor([1]*ingore_num*text_embeddings.size(0),
                                         device=text_embeddings.device).view(text_embeddings.size(0), -1)
            attentionmask = torch.cat([attentionmask, attention_mask], dim=1)
        else:
            attention_mask = None

        if labels is not None:
            labels_ = torch.tensor([-100]*ingore_num*text_embeddings.size(0),
                                   device=text_embeddings.device).view(text_embeddings.size(0), -1)
            labels = torch.cat([labels_, labels], dim=1)
        else:
            labels = None

          # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "inputs_embeds": text_embeddings,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attentionmask,
                "labels": labels,
            }
        )
        return model_inputs

    def encode_audio(self, audios):
        audio_features = self.audio_encoder.encoder(audios)
        return audio_features[0]

    def encode_image(self, images):

        image_features = self.visual_projection(
            self.image_encoder.vision_model(images)[0])[:, 1:, :]

        return image_features
