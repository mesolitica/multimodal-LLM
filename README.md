# multimodal-LLM

Multi-Modal Language Modeling with Image, Audio and Text Integration, included multi-images and multi-audio in a single multiturn.

## The goal

1. We want in one turn, user can feed multi-images or multi-audio.
2. In multi-turn, at N position, user can feed an image / an audio, and at N + k position, user can feed another an image or an audio.

## dataset

We are from Malaysia, so our dataset focused on Malaysian context, but we will train bi-lingual.

### Audio instruction

Uploaded at https://huggingface.co/collections/mesolitica/audio-malaysian-llm-6590b69ee7c71d6d9e209104

1. We crawled youtube videos, and convert to mp3.
2. Pseudolabel using Whisper Large V3, postfilter based on score threshold.
3. Use Mixtral to generate multiturn.

### Vision instruction

Uploaded at https://huggingface.co/collections/mesolitica/vision-malaysian-llm-653a16214037a1bc4417eb3a

## how-to

All pretrained and finetuned models published at https://huggingface.co/collections/mesolitica/multimodal-malaysian-llm-65c6f893e03f78fa9e5c8859, instructions inside each model cards README.

### 1. Vision Alignment

1. Prepare vision alignment dataset first [prepare-dataset/prepare-vision-alignment.ipynb](prepare-dataset/prepare-vision-alignment.ipynb), this is the exact alignment dataset used by LLAVA and we also included malay translated dataset.

2. Train vision alignment, we trained on 3 different models,

- [vision-only/run-vision-alignment-qwen0.5.sh](vision-only/run-vision-alignment-qwen0.5.sh)
- [vision-only/run-vision-alignment-tinyllama.sh](vision-only/run-vision-alignment-tinyllama.sh)
- [vision-only/run-vision-alignment-mistral.sh](vision-only/run-vision-alignment-mistral.sh)

3. Prepare instruction dataset, must follow the order,

- [prepare-dataset/prepare-relationship-vision.ipynb](prepare-dataset/prepare-relationship-vision.ipynb), synthetic dataset to compare relationship between 2 images, available for both english and malay.
- [prepare-dataset/prepare-llava.ipynb](prepare-dataset/prepare-llava.ipynb), original instruction dataset from LLAVA, included malay translated.
- [prepare-dataset/prepare-combine-vision.ipynb](prepare-dataset/prepare-combine-vision.ipynb), random combine relationship vision dataset and llava to become multiturn with multi-images in one session.
- [prepare-dataset/combine-mosaic-vision.ipynb](prepare-dataset/combine-mosaic-vision.ipynb), convert all to become mosaic format.

4. Finetune on instruction dataset,

- [vision-only/run-vision-qwen0.5.sh](vision-only/run-vision-qwen0.5.sh)
- [vision-only/run-vision-tinyllama.sh](vision-only/run-vision-tinyllama.sh)
- [vision-only/run-vision-mistral.sh](vision-only/run-vision-mistral.sh)

**But we only finetuned vision instruction dataset for Qwen 0.5B and TinyLLama only, Mistral we intended for vision and audio**.

### 2. Audio Alignment

1. Prepare audio alignment dataset first [prepare-dataset/prepare-audio-alignment.ipynb](prepare-dataset/prepare-audio-alignment.ipynb), this is pseudolabel from Whisper Large V3 and first assistant answer.

2. Train audio alignment, we trained on 2 different models,

- [audio-only/run-audio-alignment-tinyllama.sh](audio-only/run-audio-alignment-tinyllama.sh)
- [audio-only/run-audio-alignment-mistral.sh](audio-only/run-audio-alignment-mistral.sh)

3. Prepare instruction dataset, must follow the order,

- [prepare-dataset/prepare-audio.ipynb](prepare-dataset/prepare-audio.ipynb), multiturn generated using Mixtral.

### 3. Vision and Audio finetuning

1. You must combine pretrained vision and audio alignment models first,

- [combine-tinyllama.ipynb](combine-tinyllama.ipynb).
- [combine-mistral.ipynb](combine-mistral.ipynb).

2. Prepare dataset, [prepare-dataset/combine-mosaic.ipynb](prepare-dataset/combine-mosaic.ipynb), this to combine vision and audio dataset in one mosaic dataset, we only trained on 50% of the dataset due to lack of resources.

3. Finetune on instruction dataset,

- [train-tinyllama.sh](train-tinyllama.sh)
- [train-mistral.sh](train-mistral.sh)