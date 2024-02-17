seed=55
dataset_name=${1:-"aokvqa"}
type="adversarial"
model=LLaVA-RLHF-13b-v1.5-336 #LLaVA-RLHF-7b-v1.5-224 LLaVA-RLHF-13b-v1.5-336
model_path=${4:-"/mnt/workspace/xue.w/yf/checkpoint/${model}/sft_model"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}
save_dir=${2:-'default'}
temp=${3:-0.2}

echo $dataset_name $save_dir $temp
# nohup bash /mnt/workspace/xue.w/yf/VCD/experiments/eval/calibrate/run_llava.sh > pope_llava13B_aokvqa_all_temp02_naive.out 2>&1 &
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/mnt/workspace/xue.w/yf/data/coco/val2014
else
  image_folder=/mnt/workspace/xue.w/yf/data/gqa/images
fi

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_naive.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  > llava_best_$save_dir${dataset_name}_pope_${type}_seed${seed}.out

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_cd.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_cd > llava_best_cd_$save_dir_${dataset_name}_pope_${type}_seed${seed}.out

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > llava_best_dd_$save_dir_${dataset_name}_pope_${type}_seed${seed}.out


type="popular"

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_naive.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  > llava_best_$save_dir${dataset_name}_pope_${type}_seed${seed}.out

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_cd.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_cd > llava_best_cd_$save_dir_${dataset_name}_pope_${type}_seed${seed}.out

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > llava_best_dd_$save_dir_${dataset_name}_pope_${type}_seed${seed}.out


type="random"
python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_naive.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  > llava_best_$save_dir${dataset_name}_pope_${type}_seed${seed}.out

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_cd.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_cd > llava_best_cd_$save_dir_${dataset_name}_pope_${type}_seed${seed}.out

python eval/llava_calibrate_rlhf.py \
--model-path ${model_path} \
--question-file /mnt/workspace/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step  --temperature $temp \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > llava_best_dd_$save_dir_${dataset_name}_pope_${type}_seed${seed}.out