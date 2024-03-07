# KULLM2
# 13번에서 돌림 (A6000 세팅)

# upstage/SOLAR-10.7B-v1.0
# Upstage/SOLAR-10.7B-Instruct-v1.0

# IMPROVED mistral oneprompt sv18
export WANDB_PROJECT="KULLM2"
export WANDB_NAME="solar-notinstruct-ko"
EPOCH=2
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --mode $WANDB_NAME \
    --base_model 'upstage/SOLAR-10.7B-v1.0' \
    --output_dir "/mnt/raid6/potatowook/KULLM2-ckpt/"$WANDB_NAME \
    --data_mixture "[kullm2_alpaca_gpt4, kullm2_xp3x_filtered_gpt4, kullm2_dolly_gpt4, kullm2_aya, koalpaca_v1_1]" \
    --vram_available "82GB" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --warmup_steps 50 \
    --logging_steps 4 \
    --lr_scheduler_type "cosine" \
    --prompt_template_name solar