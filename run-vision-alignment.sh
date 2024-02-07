WANDB_PROJECT=vision-alignment \
python3 vision_alignment.py \
--model_name_or_path mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v3 \
--image_encoder_name_or_path google/siglip-base-patch16-384 \
--train_file mosaic-vision \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 1 \
--num_train_epochs 3 \
--save_strategy "steps" \
--save_steps 200 \
--save_total_limit 2 \
--learning_rate 1e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--logging_steps 1 \
--do_train \
--bf16 True \
--gradient_checkpointing True \
--output_dir "vision-alignment" \
--use_flash_attention2 True \
--block_size 8192