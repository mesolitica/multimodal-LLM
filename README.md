# multimodal-LLM

Multi-Modal Language Modeling with Image, Audio and Text Integration, included multi-images and multi-audio in a single multiturn.

## The goal

1. We want in one turn, user can feed multi-images or multi-audio.
2. In multi-turn, at N position, user can feed an image / an audio, and at N + k position, user can feed another an image or an audio.

## dataset

We are from Malaysia, so our dataset focused on Malaysian context.

### Audio instruction

Uploaded at https://huggingface.co/collections/mesolitica/audio-malaysian-llm-6590b69ee7c71d6d9e209104

1. We crawled youtube videos, and convert to mp3.
2. Pseudolabel using Whisper Large V3, postfilter based on score threshold.
3. Use Mixtral to generate multiturn.

### Vision instruction

Uploaded at https://huggingface.co/collections/mesolitica/vision-malaysian-llm-653a16214037a1bc4417eb3a