NCCL_SOCKET_IFNAME=eth0:120000 WANDB_PROJECT=multimodal-tinyllama \
deepspeed run.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path ./combine-tinyllama \
--audio_encoder_name_or_path mesolitica/malaysian-whisper-small \
--image_encoder_name_or_path google/siglip-base-patch16-384 \
--train_file mosaic-multimodal-v2 \
--per_device_train_batch_size 8 \
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
--output_dir "multimodal-tinyllama" \
--use_flash_attention2 True \
--block_size 8192