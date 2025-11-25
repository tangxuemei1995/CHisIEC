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

torchrun --nnodes 1 --nproc_per_node 1 main.py \
    --do_train \
    --train_file /ceph/home/jun01/tangxuemei/glm2/data/chapter/train_re_glm.json \
    --validation_file  /ceph/home/jun01/tangxuemei/glm2/data/chapter/dev_re_glm.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /ceph/home/jun01/tangxuemei/glm/glm2 \
    --output_dir output/chapter/re-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 300 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --logging_steps 10 \
    --num_train_epochs 30 \
    --save_steps 200 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4

