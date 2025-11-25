#! /bin/bash

#SBATCH --partition=IAI_SLURM_A100
#SBATCH --job-name=acr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 48:00:00

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0  python3 main.py \
    --do_train \
    --train_file /workspace/tangxuemei/chatglm2/data/coling/train_re_glm.json \
    --validation_file  /workspace/tangxuemei/chatglm2/data/coling/train_dev_re_glm.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /workspace/tangxuemei/chatglm2/glm2 \
    --output_dir output/re_coling_10-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 300 \
    --max_target_length 256 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4

