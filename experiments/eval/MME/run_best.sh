#!/bin/bash

# size=13
# model=liuhaotian/llava-v1.5-${size}b

# python eval/MME/run_calibrate.py \
#     --model-path $model \
#     --question-file ./eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
#     --answers-file ./eval/MME/answers/llavav1.5-${size}b-use-dd-both-setting.jsonl \
#     --conv-mode vicuna_v1 --use_dd_unk --use_dd > mme_dd_${size}b_both_best.out 2>&1 & #doing

# nohup python eval/MME/run_calibrate.py \
#     --model-path $model \
#     --question-file ./eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
#     --answers-file ./eval/MME/answers/llavav1.5-${size}b-use-cd-setting.jsonl \
#     --conv-mode vicuna_v1 --use_cd > mme_dd_${size}b_vcd_best.out 2>&1 &

# python eval/MME/run_best.py \
#     --model-path $model \
#     --question-file ./eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
#     --answers-file ./eval/MME/answers/llavav1.5-${size}b-naive-setting.jsonl \
#     --conv-mode vicuna_v1 > mme_dd_${size}b_naive_best.out 2>&1 &




size=VL-Chat
model=Qwen/Qwen-$size
# nohup bash /mnt/workspace/xue.w/yf/VCD/experiments/eval/MME/run_best.sh > mme_vl-chat.out 2>&1 &

echo $model

CUDA_VISIBLE_DEVICES=6 python eval/MME/run_best_qwen.py \
    --model-path $model \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/${size}-use-dd-both-setting.jsonl \
    --conv-mode vicuna_v1 --use_dd_unk --use_dd > mme_${size}_both_best.out 2>&1 & #doing

# CUDA_VISIBLE_DEVICES=7 python eval/MME/run_best_qwen.py \
#     --model-path $model \
#     --question-file ./eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
#     --answers-file ./eval/MME/answers/${size}-use-cd-setting.jsonl \
#     --conv-mode vicuna_v1 --use_cd > mme_${size}_vcd_best.out 2>&1 &

# CUDA_VISIBLE_DEVICES=7 python eval/MME/run_best_qwen.py \
#     --model-path $model \
#     --question-file ./eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
#     --answers-file ./eval/MME/answers/${size}-naive-setting.jsonl \
#     --conv-mode vicuna_v1 > mme_${size}_naive_best.out 2>&1 &