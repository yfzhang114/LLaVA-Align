#!/bin/bash

size=13
model=liuhaotian/llava-v1.5-${size}b

# using SFT model
root=your_cache_dir
model=liuhaotian/llava-v1.5-7b # $root/LLaVA-RLHF-7b-v1.5-224/sft_model

# naive
python eval/MME/run_llava.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llavav1.5-${size}b-use-dd-both-setting.jsonl \
    --conv-mode vicuna_v1 --use_dd_unk --use_dd > mme_dd_${size}b_both_best.out 2>&1 & #doing

# vcd
nohup python eval/MME/run_llava.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llavav1.5-${size}b-use-cd-setting.jsonl \
    --conv-mode vicuna_v1 --use_cd > mme_dd_${size}b_vcd_best.out 2>&1 &

# vdd & sampling
python eval/MME/run_llava.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llavav1.5-${size}b-naive-setting.jsonl \
    --conv-mode vicuna_v1 > mme_dd_${size}b_naive_best.out 2>&1 &
