CUDA_VISIBLE_DEVICES=0
num_gpu=1

# polyglot-ko-12.8B copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=EleutherAI/polyglot-ko-12.8b \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# KULLM (3000) copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=/mnt/raid6/potatowook/kullm-enhance-ckpt/kullm-enhance1_3/checkpoint-2700,tokenizer=nlpai-lab/kullm-polyglot-12.8b-v2 \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# KULLM (1000) copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=/mnt/raid6/potatowook/kullm-enhance-ckpt/kullm-enhance1_2/checkpoint-2700,tokenizer=nlpai-lab/kullm-polyglot-12.8b-v2 \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# KULLM (base) copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=nlpai-lab/kullm-polyglot-12.8b-v2 \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# beomi/llama-2-ko-7b copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=beomi/llama-2-ko-7b \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# beomi/KoAlpaca-Polyglot-12.8B copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=beomi/KoAlpaca-Polyglot-12.8B \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# KoAlpaca-Polyglot-5.8B copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=beomi/KoAlpaca-Polyglot-5.8B \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# beomi/llama-2-koen-13b copa, hellaswag, sentineg, boolq
#torchrun --nproc_per_node=$num_gpu test_do_evalharness.py \
#--model hf \
#--model_args pretrained=beomi/llama-2-koen-13b \
#--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
#--batch_size auto

# for debugging
python test_do_evalharness.py \
--model hf \
--model_args pretrained=EleutherAI/polyglot-ko-1.3b \
--tasks kobest_copa,kobest_boolq,kobest_sentineg,kobest_hellaswag \
--batch_size auto
