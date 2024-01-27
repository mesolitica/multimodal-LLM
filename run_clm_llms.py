#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.
import os
import logging
import math
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch
import datasets
import evaluate
import torch
from datasets import load_dataset
import librosa
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import LocalDataset
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import copy
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor, CLIPConfig, LlamaConfig, WhisperConfig, WhisperModel, LlamaModel, LlamaTokenizer
from transformers import AutoConfig, AutoModel
from transformers import Trainer
import argparse
import random
import numpy as np
import json
from collections.abc import Mapping
from PIL import Image
import os
from tqdm import tqdm, trange
import os
from modeling import MM_LLMs, MM_LLMs_Config
import torch
from typing import Mapping, Union, List, Dict


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


class DataCollator():

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, features):

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        batch = {}
        bs = len(features)
        first = features[0]

        batch['audio_index'] = torch.tensor([], dtype=torch.int)
        batch['image_index'] = torch.tensor([], dtype=torch.int)

        for index, feature in enumerate(features):
            local_index = index % (bs // torch.cuda.device_count()) if bs > 1 else index % (bs)
            if feature['audios'] is not None:
                batch['audio_index'] = torch.cat([batch['audio_index'], torch.tensor(
                    [local_index] * len(feature['audios']), dtype=torch.int)])

            if feature['images'] is not None:
                batch['image_index'] = torch.cat([batch['image_index'], torch.tensor(
                    [local_index] * len(feature['images']), dtype=torch.int)])

        for k, v in first.items():

            if k not in ("audios", "images") and not isinstance(v, str):
                if v is None:
                    batch[k] = None
                elif isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features]).squeeze(1)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features])).squeeze(1)
            elif k in ("audios", "images"):
                if v is None:
                    batch[k] = None
                else:
                    batch[k] = torch.cat([f[k] for f in features if f[k] is not None])

        batch['image_starts'] = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids('<image>')] * bs, dtype=torch.int)
        batch['image_ends'] = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids('</image>')] * bs, dtype=torch.int)
        batch['audio_starts'] = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids('<audio>')] * bs, dtype=torch.int)
        batch['audio_ends'] = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids('</audio>')] * bs, dtype=torch.int)

        return batch


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    audio_encoder_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    image_encoder_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models).")},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )

    n_frames: Optional[int] = field(
        default=6,
        metadata={
            "help": "The number of frames for encoding a video."
        },
    )
    attention_heads: Optional[int] = field(
        default=220,
        metadata={
            "help": "The number of attention heads used in multi-head-attention."
        },
    )

    image_conv_kernel: Optional[int] = field(
        default=48,
        metadata={
            "help": "The size of the convolutional kernel for the image stream."
        },
    )
    image_conv_stride: Optional[int] = field(
        default=36,
        metadata={
            "help": "The stride of the convolutional kernel for the image stream."
        },
    )
    use_flash_attention2: Optional[bool] = field(
        default=False
    )
    audio_conv_kernel: Optional[int] = field(
        default=240,
        metadata={
            "help": "The size of the convolutional kernel for the audio stream."
        },
    )
    audio_conv_stride: Optional[int] = field(
        default=220,
        metadata={
            "help": "The stride of the convolutional kernel for the audio stream."
        },
    )

    freeze_multi_modal_encoder: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the parameters of multi-modal encoders during training.)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}, )
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set.")}, )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set.")}, )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.padding_side = "right"
    tokenizer.chat_template = """{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{% if message['content'] is not none and message['content'].strip() != '' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{% if messages[loop.index0 - 1]['content'] is not none and messages[loop.index0 - 1]['content'].strip() != '' and message['content'] is not none and message['content'].strip() != '' %}{{ message['content'] + eos_token }}{% endif %}{% else %}{{ print('Unexpected role encountered:', message['role']) }}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""
    tokenizer.add_tokens(["<image>", "</image>", "<audio>", "</audio>"])

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level
        # at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load model
    image_config = AutoConfig.from_pretrained(model_args.image_encoder_name_or_path)
    audio_config = AutoConfig.from_pretrained(model_args.audio_encoder_name_or_path)
    llm_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model_config = MM_LLMs_Config(
        n_frames=model_args.n_frames,
        attention_heads=model_args.attention_heads,
        image_conv_kernel=model_args.image_conv_kernel,
        image_conv_stride=model_args.image_conv_stride,
        audio_conv_kernel=model_args.audio_conv_kernel,
        audio_conv_stride=model_args.audio_conv_stride,
        image_config=image_config, audio_config=audio_config, llm_config=llm_config)

    # load model separately
    model = MM_LLMs(config=model_config)

    image_processor = AutoProcessor.from_pretrained(model_args.image_encoder_name_or_path)
    audio_processor = AutoProcessor.from_pretrained(model_args.audio_encoder_name_or_path)

    model.image_encoder = model.image_encoder.from_pretrained(model_args.image_encoder_name_or_path)
    model.audio_encoder = model.audio_encoder.from_pretrained(model_args.audio_encoder_name_or_path)
    model.llm = model.llm.from_pretrained(model_args.model_name_or_path,
                                          use_flash_attention_2=model_args.use_flash_attention2,
                                          torch_dtype=torch.bfloat16)

    model.llm.resize_token_embeddings(len(tokenizer))

    model.config.llm_config.vocab_size = len(tokenizer)

    # freeze encoder model
    for param in model.image_encoder.parameters():
        param.requires_grad = False
        model.image_encoder._requires_grad = False

    for param in model.audio_encoder.parameters():
        param.requires_grad = False
        model.audio_encoder._requires_grad = False

    class ListOfDict(Encoding):
        def encode(self, obj: List[dict]) -> bytes:
            # Convert the list of dictionaries to a JSON-encoded string
            json_str = json.dumps(obj)
            return json_str.encode('utf-8')

        def decode(self, data: bytes) -> List[dict]:

            # Decode the JSON-encoded string back to a list of dictionaries
            json_str = data.decode('utf-8')
            return json.loads(json_str)

    # Register the custom encoding for 'list_of_dict'
    _encodings['list_of_dict'] = ListOfDict

    class MMDataset(torch.utils.data.Dataset):

        def __init__(self, folder):
            if folder.endswith('.json'):
                with open(folder) as fopen:
                    self.dataset = json.load(fopen)
            else:
                self.dataset = LocalDataset(folder)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            audio_list = []
            image_list = []

            for x in data['filename']:
                if x.endswith('.mp3'):
                    audio, _ = librosa.load(f'../filter-audio/{x.split("/")[-1]}', sr=16000)

                    audio_features = audio_processor(
                        audio, sampling_rate=16_000, return_tensors='pt')

                    audio_list.append(audio_features['input_features'])

                elif x.endswith('.jpg'):
                    image = Image.open(x)

                    image_output = image_processor(
                        images=image, return_tensors='pt')['pixel_values']

                    image_list.append(image_output)

            full_text = tokenizer.apply_chat_template(data['conversations'], tokenize=False)

            outputs = tokenizer(
                full_text,
                return_tensors='pt',
                truncation=True,
                padding="max_length",
                max_length=4096,
                return_overflowing_tokens=False,
                return_length=False)

            outputs['labels'] = outputs['input_ids'].clone()

            outputs['audios'] = torch.cat(audio_list, dim=0) if audio_list else None
            outputs['images'] = torch.cat(image_list, dim=0) if image_list else None

            return outputs

        def __len__(self):
            return len(self.dataset)

    data_collator = DataCollator(tokenizer=tokenizer)

    train_dataset = MMDataset(data_args.train_file)

    if training_args.local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval
        and not is_torch_tpu_available() else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Saves the tokenizer too for easy upload
        trainer.save_model(output_dir=training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        audio_processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
