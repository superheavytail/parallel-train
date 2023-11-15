#     --fp16 \
#    --weight_decay 0. \

#    --train_on_inputs \
# EleutherAI/polyglot-ko-1.3b
# EleutherAI/polyglot-ko-5.8b
# EleutherAI/polyglot-ko-12.8b
export WANDB_PROJECT="hclt"
export WANDB_NAME="kullm-base1"
EPOCH=1
LR=5e-06

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_polyglot.py \
    --debug true \
    --mode $WANDB_NAME \
    --base_model 'EleutherAI/polyglot-ko-1.3b' \
    --output_dir "/mnt/raid6/potatowook/hclt/ckpt/"$WANDB_NAME \
    --do_peft False \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 512 \
    --val_set_size 0 \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --eval_steps 800 \
    --save_steps 100 \
    --lr_scheduler_type "cosine" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "[query_key_value, dense, dense_h_to_4h, dense_4h_to_h]" \
    --group_by_length \
    --add_eos_token \
    --prompt_template_name kullm \


