seed=55
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

CUDA_VISIBLE_DEVICES=3 nohup python eval/qwenvl_sampling.py \
--model-path Qwen/Qwen-VL \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/sampling/qwen/qwen_${dataset_name}_pope_${type}_answers_seed${seed}_setting.jsonl \
--noise_step $noise_step \
--seed $seed  > qwen_sampling${dataset_name}_pope_${type}_answers_seed${seed}.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python eval/instructblip_sampling.py \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/sampling/blip/blip_${dataset_name}_pope_${type}_answers_seed${seed}_setting.jsonl \
--noise_step $noise_step \
--seed $seed  > blip_sampling${dataset_name}_pope_${type}_answers_seed${seed}.out 2>&1 &

