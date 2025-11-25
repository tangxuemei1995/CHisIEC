#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=cbdb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 48:00:00

SYSTEM_PROMPT='你是一个语义关系分类的工具'
# SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.
MODEL_PATH=/ceph/home/jun01/tangxuemei/Alpaca2/scripts/alpaca_re-combined_30_lora/ggml-model-q6_K.bin
# FIRST_INSTRUCTION=$2
cat  /ceph/home/jun01/tangxuemei/Alpaca2/scripts/inference/data/test_sheng.txt | while read line
do 
    ./main -m "$MODEL_PATH" \
    --color -i -c 1024 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 -eps 1e-5 \
    --in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]' -p \
    "[INST] <<SYS>>
    $SYSTEM_PROMPT
        <</SYS>>

    $line [/INST]"
done