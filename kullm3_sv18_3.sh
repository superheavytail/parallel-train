# KULLM3

# upstage/SOLAR-10.7B-v1.0
# Upstage/SOLAR-10.7B-Instruct-v1.0

# without xp3x
export WANDB_PROJECT="KULLM3"
export WANDB_NAME="kosolar-ko-en-noxP3x"
EPOCH=1
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --base_model 'beomi/OPEN-SOLAR-KO-10.7B' \
    --output_dir "/mnt/raid6/potatowook/KULLM2-ckpt/"$WANDB_NAME \
    --data_mixture "[kullm2_alpaca_gpt4, kullm2_dolly_gpt4, kullm2_aya, koalpaca_v1_1, alpaca_gpt4]" \
    --vram_available "82GB" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --warmup_steps 100 \
    --logging_steps 4 \
    --lr_scheduler_type "cosine" \
    --prompt_template_name solar \
    --bf16