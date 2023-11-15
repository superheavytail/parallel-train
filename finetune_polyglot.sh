#     --fp16 \
#    --weight_decay 0. \

#    --train_on_inputs \
# EleutherAI/polyglot-ko-1.3b
# EleutherAI/polyglot-ko-5.8b
# EleutherAI/polyglot-ko-12.8b

## ENHANCED (max_example=3000) sv18
#export WANDB_PROJECT="kullm-enhance"
#export WANDB_NAME="kullm-enhance1_3"
#EPOCH=2
#LR=5e-06
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune_polyglot.py \
#    --debug False \
#    --mode $WANDB_NAME \
#    --base_model 'EleutherAI/polyglot-ko-12.8b' \
#    --output_dir "/mnt/raid6/potatowook/kullm-enhance-ckpt/"$WANDB_NAME \
#    --data_mixture "[kullm_v2, klue, kobest]" \
#    --max_example 3000 \
#    --vram_available "82GB" \
#    --per_device_train_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 512 \
#    --val_set_size 0 \
#    --warmup_steps 50 \
#    --logging_steps 2 \
#    --eval_steps 800 \
#    --save_steps 300 \
#    --lr_scheduler_type "cosine" \
#    --group_by_length \
#    --add_eos_token

## ENHANCED (max_example=1000) sv14
#export WANDB_PROJECT="kullm-enhance"
#export WANDB_NAME="kullm-enhance1_2"
#EPOCH=2
#LR=5e-06
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune_polyglot.py \
#    --debug False \
#    --mode $WANDB_NAME \
#    --base_model 'EleutherAI/polyglot-ko-12.8b' \
#    --output_dir "/mnt/raid6/potatowook/kullm-enhance-ckpt/"$WANDB_NAME \
#    --data_mixture "[kullm_v2, klue, kobest]" \
#    --max_example 1000 \
#    --vram_available "82GB" \
#    --per_device_train_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 512 \
#    --val_set_size 0 \
#    --warmup_steps 50 \
#    --logging_steps 2 \
#    --eval_steps 800 \
#    --save_steps 300 \
#    --lr_scheduler_type "cosine" \
#    --group_by_length \
#    --add_eos_token
#
# ENHANCED (max_example=300) sv9
export WANDB_PROJECT="kullm-enhance"
export WANDB_NAME="kullm-enhance1_1"
EPOCH=2
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune_polyglot.py \
    --debug False \
    --mode $WANDB_NAME \
    --base_model 'EleutherAI/polyglot-ko-12.8b' \
    --output_dir "/mnt/raid6/potatowook/kullm-enhance-ckpt/"$WANDB_NAME \
    --data_mixture "[kullm_v2, klue, kobest]" \
    --max_example 300 \
    --vram_available "48GB" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 512 \
    --val_set_size 0 \
    --warmup_steps 50 \
    --logging_steps 2 \
    --eval_steps 800 \
    --save_steps 482 \
    --lr_scheduler_type "cosine" \
    --group_by_length \
    --add_eos_token
#
## BASELINE(PURE KULLM) sv17
#export WANDB_PROJECT="kullm-enhance"
#export WANDB_NAME="kullm-base1"
#EPOCH=1
#LR=5e-06
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune_polyglot.py \
#    --debug False \
#    --mode $WANDB_NAME \
#    --base_model 'EleutherAI/polyglot-ko-12.8b' \
#    --output_dir "/mnt/raid6/potatowook/kullm-enhance-ckpt/"$WANDB_NAME \
#    --data_mixture "[kullm_v2]" \
#    --max_example 300 \
#    --vram_available "48GB" \
#    --per_device_train_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 512 \
#    --val_set_size 0 \
#    --warmup_steps 50 \
#    --logging_steps 2 \
#    --eval_steps 800 \
#    --save_steps 300 \
#    --lr_scheduler_type "cosine" \
#    --group_by_length \
#    --add_eos_token

