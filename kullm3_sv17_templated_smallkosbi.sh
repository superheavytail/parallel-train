# KULLM3 (bsz 8)

# upstage/SOLAR-10.7B-v1.0
# Upstage/SOLAR-10.7B-Instruct-v1.0
export WANDB_PROJECT="KULLM3"
export WANDB_NAME="solar-instruct-ko-en-noxP3x-templated-smallkosbi"
EPOCH=1
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --base_model 'Upstage/SOLAR-10.7B-Instruct-v1.0' \
    --output_dir "/mnt/raid6/potatowook/KULLM3-ckpt/"$WANDB_NAME \
    --data_mixture "[kullm3_alpaca_gpt4, kullm3_dolly_gpt4, kullm3_aya, koalpaca_v1_1, alpaca_gpt4, kullm3_personal_info, kullm3_square_gpt4_sampled]" \
    --vram_available "48GB" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --warmup_steps 50 \
    --logging_steps 4 \
    --lr_scheduler_type "cosine" \
    --prompt_template_name solar \
    --apply_chat_template \
    --use_wandb