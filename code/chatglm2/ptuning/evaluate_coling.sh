#! /bin/bash

#SBATCH --partition=IAI_SLURM_A100
#SBATCH --job-name=acr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 48:00:00

PRE_SEQ_LEN=128
CHECKPOINT=re_coling_10-128-2e-2
STEP=2100
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0  python3 main.py \
    --do_predict \
    --validation_file /workspace/tangxuemei/chatglm2/data/coling/train_dev_re_glm.json \
    --test_file /workspace/tangxuemei/chatglm2/data/coling/test_re_glm.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column answer \
    --model_name_or_path /workspace/tangxuemei/chatglm2/glm2 \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 300 \
    --max_target_length 256 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

