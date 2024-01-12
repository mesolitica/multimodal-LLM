WANDB_PROJECT_DISABLED=true \
  ~/.local/bin/deepspeed run_clm_llms.py \
    --attention_heads 8 \
    --deepspeed ds_config_zero3.json \
    --model_name_or_path mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v3 \
    --audio_encoder_name_or_path mesolitica/malaysian-whisper-tiny \
    --image_encoder_name_or_path openai/clip-vit-base-patch16 \
    --train_file /home/ubuntu/resepichenom.com-multiturn/chat.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 3 \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --block_size 512 \
    --do_train \
    --fp16 True \
    --gradient_checkpointing False \
    --output_dir MM-LLMs 