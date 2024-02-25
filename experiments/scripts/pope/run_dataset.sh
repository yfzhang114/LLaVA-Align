#!/bin/bash

datasets=("aokvqa" "gqa" "coco")
temp=1.0

if [ ! -d "./output/calibrate_best_sample/${save_dir}" ];then
mkdir ./output/calibrate_best_sample/${save_dir}
echo "success create dir"
else
echo "dir exists"
fi

=
for index in "${!datasets[@]}"
do
    dataset="${datasets[$index]}"
    device=$((index)) 

    #llava results
    model_name=llava_7b
    save_dir=${model_name}_temp${temp}
    command="bash ./scripts/pope/run_llava.sh $dataset $save_dir $temp"

    # qwen results
    model_name=qwen-chat
    save_dir=${model_name}_temp${temp}
    command="bash ./scripts/pope/run_qwen.sh $dataset $save_dir $temp"

    echo "Running command: $command"

    nohup $command > $save_dir$dataset.log 2>&1 &
    # $command
done