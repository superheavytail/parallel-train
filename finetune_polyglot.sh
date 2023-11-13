#     --fp16 \
#    --weight_decay 0. \
#    --warmup_ratio 0.01 \
#    --train_on_inputs \
# EleutherAI/polyglot-ko-1.3b
# EleutherAI/polyglot-ko-5.8b
# EleutherAI/polyglot-ko-12.8b
export WANDB_PROJECT="hclt"
export WANDB_NAME="baseline5"
EPOCH=5
LR=5e-05
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 finetune_polyglot.py \
#    --base_model 'EleutherAI/polyglot-ko-1.3b' \
#    --data_path data/kullm-v2.jsonl \
#    --output_dir "/mnt/raid6/potatowook/hclt/ckpt/"$WANDB_NAME \
#    --prompt_template_name kullm \
#    --per_device_train_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 512 \
#    --val_set_size 0 \
#    --lora_r 8 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_target_modules "[query_key_value, dense, dense_h_to_4h, dense_4h_to_h]" \
#    --logging_steps 1 \
#    --eval_steps 800 \
#    --save_steps 5 \
#    --lr_scheduler_type "cosine" \
#    --group_by_length \
#    --mode $WANDB_NAME \
#    --add_eos_token \
#    --do_peft False \
#    --debug true

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_polyglot.py \
    --base_model 'EleutherAI/polyglot-ko-1.3b' \
    --data_path data/kullm-v2.jsonl \
    --output_dir "/mnt/raid6/potatowook/hclt/ckpt/"$WANDB_NAME \
    --prompt_template_name kullm \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 512 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "[query_key_value, dense, dense_h_to_4h, dense_4h_to_h]" \
    --logging_steps 1 \
    --eval_steps 800 \
    --save_steps 5 \
    --lr_scheduler_type "cosine" \
    --group_by_length \
    --mode $WANDB_NAME \
    --add_eos_token \
    --do_peft False \
    --debug true
