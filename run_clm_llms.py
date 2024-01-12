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
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch
import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
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
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-03-21
import copy

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor, CLIPConfig, LlamaConfig, WhisperConfig, WhisperModel, LlamaModel, LlamaTokenizer
from transformers import AutoConfig, AutoModel
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json
import pickle
import codecs
from collections.abc import Mapping
from PIL import Image
from peft import PeftModel
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm, trange
from sklearn.metrics import top_k_accuracy_score
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)
import os

from modeling import MM_LLMs, MM_LLMs_Config
import clip
import whisper
import torch


def json_load(x): return json.load(codecs.open(x, 'r', encoding='utf-8'))
def json_dump(d, p): return json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def preprocessor_new(
        sources) -> str:

    base_string = ""
    for convs in sources:
        if convs["role"] == "user":
            base_string += f"<s> [INST] {convs['content']} [/INST] "
        else:
            base_string += f"{convs['content']} </s>"

    return base_string


import torch
from typing import Mapping
import numpy as np


class DataCollator:

    tokenizer: AutoTokenizer
    audio_processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        bs = len(features)

        batch['audio_index'] = torch.tensor([], dtype=torch.int)
        batch['image_index'] = torch.tensor([], dtype=torch.int)
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

        for index, feature in enumerate(features):
            local_index = index % (bs // torch.cuda.device_count())
            if first['audios'] is not None:
                batch['audio_index'] = torch.cat([batch['audio_index'], torch.tensor(
                    [local_index] * len(feature['audios']), dtype=torch.int)])

            if first['images'] is not None:
                batch['image_index'] = torch.cat([batch['image_index'], torch.tensor(
                    [local_index] * len(feature['images']), dtype=torch.int)])

        # Handling of all other possible keys.
        for k, v in first.items():

            if k not in ("label", "audios", "images") and not isinstance(v, str):
                if v is None:
                    batch[k] = None
                elif isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features]).squeeze(1)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features])).squeeze(1)
            else:
                if v is None:
                    batch[k] = None
                else:
                    batch[k] = torch.cat([f[k] for f in features])

        batch['image_starts'] = torch.tensor(
            [tokenizer.convert_tokens_to_ids('<image>')] * bs, dtype=torch.int)
        batch['image_ends'] = torch.tensor(
            [tokenizer.convert_tokens_to_ids('</image>')] * bs, dtype=torch.int)
        batch['audio_starts'] = torch.tensor(
            [tokenizer.convert_tokens_to_ids('<audio>')] * bs, dtype=torch.int)
        batch['audio_ends'] = torch.tensor(
            [tokenizer.convert_tokens_to_ids('</audio>')] * bs, dtype=torch.int)

        return {'inputs': batch}


def data_collator(features: Mapping):

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    bs = len(features)
    batch['audio_index'] = torch.tensor([], dtype=torch.int)
    batch['image_index'] = torch.tensor([], dtype=torch.int)

    # Special handling.
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(
            first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    for k, v in first.items():

        if k not in ("label", "audios", "images") and not isinstance(v, str):
            if v is None:
                batch[k] = None
            elif isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features]).squeeze(1)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features])).squeeze(1)
        else:
            if v is None:
                batch[k] = None
            else:
                batch[k] = torch.cat([f[k] for f in features])

    for index, feature in enumerate(features):
        local_index = index % (bs // torch.cuda.device_count())
        if first['audios'] is not None:
            batch['audio_index'] = torch.cat([batch['audio_index'], torch.tensor(
                [local_index] * len(feature['audios']), dtype=torch.int)])

        if first['images'] is not None:
            batch['image_index'] = torch.cat([batch['image_index'], torch.tensor(
                [local_index] * len(feature['images']), dtype=torch.int)])

    # batch['image_starts'] = torch.tensor([tokenizer.convert_tokens_to_ids('<image>')] * bs, dtype=torch.int)
    # batch['image_ends'] = torch.tensor([tokenizer.convert_tokens_to_ids('</image>')] * bs, dtype=torch.int)
    # batch['audio_starts'] = torch.tensor([tokenizer.convert_tokens_to_ids('<audio>')] * bs, dtype=torch.int)
    # batch['audio_ends'] = torch.tensor([tokenizer.convert_tokens_to_ids('</audio>')] * bs, dtype=torch.int)

    return {'inputs': batch}


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

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)},
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
    video_conv_kernel: Optional[int] = field(
        default=36,
        metadata={
            "help": "The size of the convolutional kernel for the video stream."
        },
    )
    video_conv_stride: Optional[int] = field(
        default=30,
        metadata={
            "help": "The stride of the convolutional kernel for the video stream."
        },
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

    # print(training_args)

    training_args.remove_unused_columns = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_tokens(["<image>", "</image>", "<pad>", "<audio>", "</audio>"])
    tokenizer.padding_side = 'right'

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load model
    clip_config = CLIPConfig.from_pretrained(model_args.image_encoder_name_or_path)
    whisper_config = WhisperConfig.from_pretrained(model_args.audio_encoder_name_or_path)
    llm_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model_config = MM_LLMs_Config(
        n_frames=model_args.n_frames,
        attention_heads=model_args.attention_heads,
        image_conv_kernel=model_args.image_conv_kernel,
        image_conv_stride=model_args.image_conv_stride,
        audio_conv_kernel=model_args.audio_conv_kernel,
        audio_conv_stride=model_args.audio_conv_stride,
        clip_config=clip_config, whisper_config=whisper_config, llm_config=llm_config)

    # load model separately
    model = MM_LLMs(config=model_config)

    model.cuda()

    model.image_encoder.from_pretrained(model_args.image_encoder_name_or_path)
    model.audio_encoder.from_pretrained(model_args.audio_encoder_name_or_path)
    model.llm.from_pretrained(model_args.model_name_or_path, ignore_mismatched_sizes=True)
    model.llm.resize_token_embeddings(len(tokenizer))

    # freeze encoder model
    for param in model.image_encoder.parameters():
        param.requires_grad = False
        model.image_encoder._requires_grad = False

    for param in model.audio_encoder.parameters():
        param.requires_grad = False
        model.audio_encoder._requires_grad = False

    class MMDataset(torch.utils.data.Dataset):
        def __init__(self, model_args, folder):
            if folder.endswith('.json'):
                with open(folder) as fopen:
                    self.dataset = json.load(fopen)
            else:
                self.dataset = LocalDataset(folder)

            self.processor = AutoProcessor.from_pretrained(model_args.image_encoder_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

        def __getitem__(self, idx):

            data = self.dataset[idx]

            audio = librosa.load('../filter-audio/2-2027-16.mp3')[0]

            audio_features = processor(audio, sampling_rate=16_000, return_tensors='pt')

            image = Image.open(
                os.path.join(
                    '/home/ubuntu/resepichenom.com-multiturn',
                    data['image']))
            image2 = Image.open(
                os.path.join(
                    '/home/ubuntu/resepichenom.com-multiturn',
                    data['image']))

            image_output = preprocess(image).unsqueeze(0)
            image_output2 = preprocess(image2).unsqueeze(0)

            image_output = torch.cat((image_output, image_output2), 0)

            full_text = preprocessor_new(data['conversations'])

            outputs = tokenizer(full_text, return_tensors='pt', max_length=50, padding="max_length")

            outputs['labels'] = outputs['input_ids'].clone()
            outputs['audios'] = audio_features['input_features']
            outputs['images'] = image_output

            return outputs

        def __len__(self):
            return len(self.dataset)

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
        data_collator=torch_default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval
        and not is_torch_tpu_available() else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Saves the tokenizer too for easy upload
        trainer.save_model(output_dir=training_args.output_dir)

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
