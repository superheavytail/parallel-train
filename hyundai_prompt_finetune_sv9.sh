# EleutherAI/polyglot-ko-12.8b
# mistralai/Mistral-7B-Instruct-v0.2
# Upstage/SOLAR-10.7B-Instruct-v1.0

# IMPROVED KULLM sv9
export WANDB_PROJECT="hyundai-mrc-prompt"
export WANDB_NAME="mistral-prompt-nonhuman_agg"
EPOCH=1
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --mode $WANDB_NAME \
    --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
    --output_dir "/mnt/raid6/potatowook/hyundai-llm-ckpt/"$WANDB_NAME \
    --data_mixture "[kullm_v2, hyundai_nonhuman_agg_kobest, hyundai_nonhuman_agg_korquad_v1]" \
    --vram_available "48GB" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --warmup_steps 50 \
    --logging_steps 4 \
    --lr_scheduler_type "cosine" \
    --add_eos_token \
    --prompt_template_name hyundai_llm \
    --max_examples 30000