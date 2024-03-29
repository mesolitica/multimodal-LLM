WANDB_PROJECT=vision-alignment-mistral \
torchrun --nproc_per_node 8 \
-m vision_alignment \
--model_name_or_path mesolitica/malaysian-mistral-7b-32k-instructions-v4 \
--image_encoder_name_or_path google/siglip-base-patch16-384 \
--train_file mosaic-vision \
--per_device_train_batch_size 6 \
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
--output_dir "vision-alignment-mistral" \
--use_flash_attention2 True \
--block_size 2048 \
--vision_select_layer -2