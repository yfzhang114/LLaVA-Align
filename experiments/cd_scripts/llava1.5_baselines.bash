seed=53
dataset_name=${2:-"coco"}
type=${3:-"random"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}
cb_cut_weight=${7:-0.5}
cb_m_weight=${7:-0.75}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/mnt/data/xue.w/yf/data/coco/val2014
else
  image_folder=/mnt/data/xue.w/yf/data/gqa/images
fi

python ./eval/llava_naive.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_7b_seed${seed}.jsonl \
--noise_step $noise_step \
--seed $seed

python ./eval/llava_naive.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_7b_seed${seed}_cd.jsonl \
--noise_step $noise_step \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}