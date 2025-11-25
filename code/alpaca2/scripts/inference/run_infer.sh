CUDA_VISIBLE_DEVICES=0 python inference_hf.py \
    --base_model ziqingyang/chinese-alpaca-2-7b \
    --tokenizer_path ziqingyang/chinese-alpaca-2-7b \
    --data_file /ceph/home/jun01/tangxuemei/Alpaca2/scripts/inference/data/test_instruction.txt \
    --with_prompt \
    --predictions_file ./output/predictions_re_lora.json \

