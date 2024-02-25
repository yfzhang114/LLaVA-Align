#!/bin/bash

size=VL
model=Qwen/Qwen-$size
# nohup bash /mnt/workspace/xue.w/yf/VCD/experiments/eval/MME/run_best.sh > mme_vl-chat.out 2>&1 &

# echo $model

CUDA_VISIBLE_DEVICES=6 python eval/MME/run_qwen.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/${size}-use-dd-both-setting.jsonl \
    --conv-mode vicuna_v1 --use_dd_unk --use_dd > mme_${size}_both_best.out 2>&1 & #doing

CUDA_VISIBLE_DEVICES=7 python eval/MME/run_qwen.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/${size}-use-cd-setting.jsonl \
    --conv-mode vicuna_v1 --use_cd > mme_${size}_vcd_best.out 2>&1 &

CUDA_VISIBLE_DEVICES=6 python eval/MME/run_qwen.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/${size}-naive-setting.jsonl \
    --conv-mode vicuna_v1 > mme_${size}_naive_best.log 2>&1 &
