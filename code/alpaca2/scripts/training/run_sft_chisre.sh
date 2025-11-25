#! /bin/bash

#SBATCH --partition=IAI_SLURM_A100
#SBATCH --job-name=acr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 48:00:00

# nvidia-smi
# python -m bitsandbytes
lr=2e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/ceph/home/jun01/tangxuemei/llama2/alpaca2 #model path

chinese_tokenizer_path=/ceph/home/jun01/tangxuemei/llama2/alpaca2 #model path
dataset_dir=/ceph/home/jun01/tangxuemei/Alpaca2/scripts/training/data/chisre/train # data
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=1
output_dir=./output/chisre/re_epoch=5
peft_model=ziqingyang/chinese-alpaca-2-lora-7b
validation_file=/ceph/home/jun01/tangxuemei/Alpaca2/scripts/training/data/chisre/dev.json #development set

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 1 --master_port=25641 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 30 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --half_precision_backend=apex \
    --ddp_find_unused_parameters False \
    # --peft_path ziqingyang/chinese-alpaca-2-7b
