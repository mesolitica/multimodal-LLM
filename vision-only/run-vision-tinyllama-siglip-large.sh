WANDB_PROJECT=vision-tinyllama-siglip-large \
deepspeed vision_train.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path mesolitica/malaysian-tinyllama-1.1b-siglip-large-384-vision-alignment \
--image_encoder_name_or_path google/siglip-large-patch16-384 \
--train_file mosaic-multimodal-vision \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--num_train_epochs 3 \
--save_strategy "steps" \
--save_steps 50 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--warmup_ratio 0.03 \
--logging_steps 1 \
--do_train \
--bf16 True \
--gradient_checkpointing True \
--output_dir "vision-tinyllama-siglip-large" \
--use_flash_attention2 True \
--block_size 8192