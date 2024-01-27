WANDB_PROJECT_DISABLED=true \
~/.local/bin/deepspeed run_clm_llms.py \
    --deepspeed ds_config_zero2.json \
    --attention_heads 8 \
    --model_name_or_path mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v3 \
    --audio_encoder_name_or_path mesolitica/malaysian-whisper-small \
    --image_encoder_name_or_path google/siglip-base-patch16-224 \
    --train_file mosaic-multimodal-sample-3 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --do_train \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir "multimodal-tinyllama-whisper-small-siglip" 