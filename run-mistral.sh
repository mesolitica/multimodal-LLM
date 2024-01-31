WANDB_PROJECT=multimodal-mistral \
CUDA_VISIBLE_DEVICES=1 \
python3 run_clm_llms.py \
--attention_heads 8 \
--model_name_or_path mesolitica/malaysian-mistral-7b-32k-instructions-v4 \
--audio_encoder_name_or_path mesolitica/malaysian-whisper-small \
--image_encoder_name_or_path google/siglip-base-patch16-224 \
--train_file mosaic-multimodal \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 2 \
--num_train_epochs 3 \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--logging_steps 1 \
--do_train \
--bf16 True \
--gradient_checkpointing True \
--output_dir "multimodal-mistral-whisper-small-siglip" \
--use_flash_attention2 True \
--block_size 8192