# Taemin Lee gemma-2b it codeswitch

# google/gemma-2b-it
# distilgpt2

export WANDB_PROJECT="codeswitch"
export WANDB_NAME="codeswitch-gemma-2b-it-pilot-notrandom"
EPOCH=5
LR=3e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --mode $WANDB_NAME \
    --base_model 'google/gemma-2b-it' \
    --output_dir "/mnt/raid6/potatowook/taemin_ckpt/"$WANDB_NAME \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --warmup_steps 50 \
    --logging_steps 4 \
    --vram_available 82GB \
    --lr_scheduler_type "cosine" \
    --prompt_template_name gemma \
    --use_wandb

#export WANDB_PROJECT="codeswitch"
#export WANDB_NAME="codeswitch-gemma-7b-it-pilot-notrandom"
#EPOCH=5
#LR=3e-06
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
#    --debug False \
#    --mode $WANDB_NAME \
#    --base_model 'google/gemma-7b-it' \
#    --output_dir "/mnt/raid6/potatowook/taemin_ckpt/"$WANDB_NAME \
#    --per_device_train_batch_size 1 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 2048 \
#    --val_set_size 0 \
#    --warmup_steps 50 \
#    --logging_steps 4 \
#    --vram_available 82GB \
#    --lr_scheduler_type "cosine" \
#    --prompt_template_name gemma \
#    --use_wandb