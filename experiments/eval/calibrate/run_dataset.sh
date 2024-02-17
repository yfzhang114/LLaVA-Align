#!/bin/bash

# 数据集列表
datasets=("aokvqa" "gqa" "coco")
temp=1.0
model_name=llava13b_sft
# model_name=qwen-chat
save_dir=${model_name}_temp${temp}


if [ ! -d "./output/calibrate_best_sample/${save_dir}" ];then
mkdir ./output/calibrate_best_sample/${save_dir}
echo "success create dir"
else
echo "dir exists"
fi

# 循环遍历数据集
for index in "${!datasets[@]}"
do
    dataset="${datasets[$index]}"
    device=$((index))  # 使用索引加1作为CUDA_VISIBLE_DEVICES

    # 构建要执行的命令 
    command="bash eval/calibrate/run_llava.sh $dataset $save_dir $temp"

    # 输出即将执行的命令
    echo "CUDA_VISIBLE_DEVICES=$device Running command: $command"

    # 执行命令
    # CUDA_VISIBLE_DEVICES=$device nohup $command > $save_dir$dataset.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup $command > $save_dir$dataset.log 2>&1 &
    # $command
done