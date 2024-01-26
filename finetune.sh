## kullm-lbd
# === Mistral === sv12
export WANDB_PROJECT="kullm-leaderboard"
export WANDB_NAME="kullm-lbd-mistral-3ep"
EPOCH=3
LR=5e-06
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
    --debug False \
    --mode $WANDB_NAME \
    --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
    --output_dir "/mnt/raid6/potatowook/kullm-leaderboard-ckpt/"$WANDB_NAME \
    --data_mixture "[klue, kobest, ko_arc, ko_commongenv2, ko_mmlu, ko_truthfulqa]" \
    --max_example None \
    --vram_available "48GB" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --warmup_steps 200 \
    --logging_steps 4 \
    --eval_steps 800 \
    --save_steps 100 \
    --lr_scheduler_type "cosine" \
    --add_eos_token \
    --prompt_template_name kullm_leaderboard

# === SOLAR === sv15
#export WANDB_PROJECT="kullm-leaderboard"
#export WANDB_NAME="kullm-lbd-solar-3ep"
#EPOCH=3
#LR=5e-06
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
#    --debug False \
#    --mode $WANDB_NAME \
#    --base_model 'Upstage/SOLAR-10.7B-Instruct-v1.0' \
#    --output_dir "/mnt/raid6/potatowook/kullm-leaderboard-ckpt/"$WANDB_NAME \
#    --data_mixture "[klue, kobest, ko_arc, ko_commongenv2, ko_mmlu, ko_truthfulqa]" \
#    --max_example None \
#    --vram_available "48GB" \
#    --per_device_train_batch_size 2 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 2048 \
#    --val_set_size 0 \
#    --warmup_steps 200 \
#    --logging_steps 4 \
#    --eval_steps 800 \
#    --save_steps 100 \
#    --lr_scheduler_type "cosine" \
#    --add_eos_token \
#    --prompt_template_name kullm_leaderboard

## kullm-lbd-S
# === Mistral === sv12
#export WANDB_PROJECT="kullm-leaderboard"
#export WANDB_NAME="kullm-lbd-solar-3ep"
#EPOCH=3
#LR=5e-06
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finetune.py \
#    --debug False \
#    --mode $WANDB_NAME \
#    --base_model 'Upstage/SOLAR-10.7B-Instruct-v1.0' \
#    --output_dir "/mnt/raid6/potatowook/kullm-leaderboard-ckpt/"$WANDB_NAME \
#    --data_mixture "[klue, kobest, ko_arc, ko_commongenv2, ko_mmlu, ko_truthfulqa]" \
#    --max_example None \
#    --vram_available "48GB" \
#    --per_device_train_batch_size 2 \
#    --gradient_accumulation_steps 1 \
#    --num_epochs $EPOCH \
#    --learning_rate $LR \
#    --cutoff_len 2048 \
#    --val_set_size 0 \
#    --warmup_steps 200 \
#    --logging_steps 4 \
#    --eval_steps 800 \
#    --save_steps 100 \
#    --lr_scheduler_type "cosine" \
#    --add_eos_token \
#    --prompt_template_name kullm_leaderboard