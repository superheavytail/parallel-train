# Taemin Lee gemma-2b it codeswitch

# google/gemma-2b-it

export WANDB_PROJECT="KULLM_v3"
export WANDB_NAME="codeswitch-gemma-2b-it-pilot"
EPOCH=5
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --mode $WANDB_NAME \
    --base_model 'distilgpt2' \
    --output_dir "/mnt/raid6/potatowook/KULLM2-ckpt/"$WANDB_NAME \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 512 \
    --val_set_size 0 \
    --warmup_steps 50 \
    --logging_steps 4 \
    --lr_scheduler_type "cosine" \
    --prompt_template_name gemma \
    --random_switch \
    --data_mixture [kobest]