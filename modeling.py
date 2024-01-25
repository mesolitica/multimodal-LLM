from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
import math

from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers import GenerationConfig
from transformers import CLIPConfig, CLIPProcessor, CLIPModel
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
            self.image_config = CLIPConfig.from_dict(self.image_config)
        if isinstance(self.audio_config, dict):
            self.audio_config = WhisperConfig.from_dict(self.audio_config)
        if isinstance(self.llm_config, dict):
            self.llm_config = LlamaConfig.from_dict(self.llm_config)

        self.hidden_size = max(
            self.llm_config.hidden_size,
            self.image_config.projection_dim,
            self.audio_config.d_model,
        )

        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["image_config"] = self.image_config.to_dict()
        output["audio_config"] = self.audio_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['attention_heads'] = self.attention_heads
        output['image_conv_kernel'] = self.image_conv_kernel
        output['image_conv_stride'] = self.image_conv_stride
        output['audio_conv_kernel'] = self.audio_conv_kernel
        output['audio_conv_stride'] = self.audio_conv_stride
        output['hidden_size'] = self.hidden_size
        output["model_type"] = self.__class__.model_type
        return output


class MM_LLMs(PreTrainedModel):
    config_class = MM_LLMs_Config
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.image_encoder = CLIPModel(config.image_config)

        self.audio_encoder = WhisperModel(config.audio_config)

        self.llm = AutoModelForCausalLM.from_config(config.llm_config)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        self.temporal_self_attention = nn.MultiheadAttention(config.image_config.projection_dim,
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
        self.transform_image_to_hidden = nn.Linear(config.image_config.projection_dim,
                                                   config.llm_config.hidden_size)

        self.project_image = nn.Conv1d(
            config.image_config.projection_dim,
            config.image_config.projection_dim,
            kernel_size=config.image_conv_kernel,
            stride=config.image_conv_stride)
        self.project_audio = nn.Conv1d(
            config.audio_config.d_model,
            config.audio_config.d_model,
            kernel_size=config.audio_conv_kernel,
            stride=config.audio_conv_stride)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.layer_norm = nn.LayerNorm(config.image_config.projection_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.sigmoid = nn.Sigmoid()

        self.loss_fct = CrossEntropyLoss()

        self.init_weights()

    def forward(self, inputs=None):
        # """
        # :param inputs:
        #
        #             audios: B x 1
        #             images: B x 1
        #             input_ids: B x L
        #             labels: B x L
        #
        # :return: loss when training else None
        # """

        text_embeddings, attention_mask, labels = self.prepare_inputs_for_generation(inputs)

        if 'inference' in inputs and inputs['inference'] is True:

            # The code below will possibly trigger an error in :
            # https://github.com/microsoft/DeepSpeed/issues/3156 (the solution only
            # partially resolves the bug for me)
            generate_ids = self.llm.generate(
                inputs_embeds=text_embeddings,
                max_new_tokens=128,
                eos_token_id=model.llm.config.eos_token_id,
                bos_token_id=model.llm.config.bos_token_id,
                pad_token_id=model.llm.config.pad_token_id)
            return generate_ids
        outputs = self.llm(
            inputs_embeds=text_embeddings,
            attention_mask=attention_mask,
            labels=labels)

        return outputs

    def prepare_inputs_for_generation(self, inputs):

        image_features = self.encode_image(
            inputs['images']) if inputs['images'] is not None else None
        audio_features = self.encode_audio(
            inputs['audios']) if inputs['audios'] is not None else None
        embed_tokens = self.llm.model.embed_tokens
        text_embeddings = embed_tokens(inputs['input_ids'])

        token_embeddings = embed_tokens.weight.unsqueeze(0).repeat(
            text_embeddings.size(0), 1, 1).transpose(0, 1)

        ingore_num = 0

        if audio_features is not None:

            audio_index = inputs['audio_index']

            audio_starts = embed_tokens(inputs['audio_starts']).unsqueeze(1)
            audio_ends = embed_tokens(inputs['audio_ends']).unsqueeze(1)

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

            img_index = inputs['image_index']

            image_starts = embed_tokens(inputs['image_starts']).unsqueeze(1)
            image_ends = embed_tokens(inputs['image_ends']).unsqueeze(1)

            image_features = self.project_image(
                image_features.transpose(
                    1, 2).contiguous()).transpose(
                1, 2).contiguous()

            image_features = self.transform_image_to_hidden(image_features)

            max_count = most_frequent_element(img_index)

            seq_img = image_features.shape[1]
            dim = token_embeddings.shape[2]

            new_img = torch.zeros(
                (token_embeddings.shape[1],
                 seq_img * max_count,
                 dim),
                device=token_embeddings.device,
                dtype=token_embeddings.dtype)

            current_dim = 0
            for no, index in enumerate(img_index):
                if no > 0 and img_index[no - 1] == index:
                    current_dim += 1
                else:
                    current_dim = 0
                new_img[index, current_dim *
                        seq_img: (current_dim + 1) * seq_img] = image_features[no]
                last_index = img_index[0]

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

        if 'attention_mask' in inputs:
            attention_mask = torch.tensor([1]*ingore_num*text_embeddings.size(0),
                                          device=text_embeddings.device).view(text_embeddings.size(0), -1)
            attention_mask = torch.cat([attention_mask, inputs['attention_mask']], dim=1)
        else:
            attention_mask = None

        if 'labels' in inputs and inputs['labels'] is not None:
            labels = torch.tensor([-100]*ingore_num*text_embeddings.size(0),
                                  device=text_embeddings.device).view(text_embeddings.size(0), -1)
            labels = torch.cat([labels, inputs['labels']], dim=1)
        else:
            labels = None

        return text_embeddings, attention_mask, labels

    def encode_audio(self, audios):
        audio_features = self.audio_encoder.encoder(audios)
        return audio_features[0]

    def encode_image(self, images):

        image_features = self.image_encoder.visual_projection(
            self.image_encoder.vision_model(images)[0])[:, 1:, :]
        return image_features
