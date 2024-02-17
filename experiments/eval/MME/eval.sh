#!/bin/bash

nohup python eval/MME/run_calibrate.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llava-v1.5-13b-use-dd-none-setting.jsonl \
    --temperature 1.0 \
    --conv-mode vicuna_v1 --use_dd > mme_dd_none_calibrate.out 2>&1 & #doing

nohup  python eval/MME/run_calibrate.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llava-v1.5-13b-use-dd-unk-setting.jsonl \
    --temperature 1.0 \
    --conv-mode vicuna_v1 --use_dd_unk > mme_dd_unk_calibrate.out 2>&1 & #doing

CUDA_VISIBLE_DEVICES=0,1,2,3  python eval/MME/run_calibrate.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llava-v1.5-13b-use-dd-both-setting.jsonl \
    --temperature 1.0 \
    --conv-mode vicuna_v1 --use_dd_unk --use_dd > mme_dd_both_calibrate.out 2>&1 & #doing

nohup python eval/MME/run_calibrate.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llava-v1.5-13b-use-dd-both.jsonl \
    --temperature 1.0 \
    --conv-mode vicuna_v1 --use_cd > mme_dd_vcd_calibrate.out 2>&1 &

nohup python eval/MME/run_calibrate.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./eval/MME/llava_mme.jsonl \
    --image-folder /mnt/data/xue.w/yf/data/MME_Benchmark \
    --answers-file ./eval/MME/answers/llava-v1.5-13b-naive.jsonl \
    --temperature 1.0 \
    --conv-mode vicuna_v1 > mme_dd_naive_calibrate.out 2>&1 &

python eval/MME/convert_answer_to_mme.py --experiment llava-v1.5-7b
find /mnt/data/xue.w/yf/VCD/experiments/eval/MME/answers -type f -exec sh -c 'filename=$(basename "$0" .jsonl); python eval/MME/convert_answer_to_mme.py --experiment "$filename"' {} \; # extract all files and convert all of them to MME format

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b
