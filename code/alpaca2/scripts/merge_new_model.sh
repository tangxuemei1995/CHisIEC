python merge_llama2_with_chinese_lora_low_mem.py \
    --base_model /ceph/home/jun01/tangxuemei/llama2/alpaca2/ \
    --lora_model /ceph/home/jun01/tangxuemei/Alpaca2/scripts/training/output/chisre/re_epoch=30/sft_lora_model \
    --output_type huggingface \
    --output_dir /ceph/home/jun01/tangxuemei/Alpaca2/alpaca_chisre_re_epoch=30