seed=55
dataset_name=${2:-"coco"}
type=${3:-"adversarial"}
model_path=${4:-"Qwen/Qwen-VL"}
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

CUDA_VISIBLE_DEVICES=0 python eval/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate/qwen/qwen_${dataset_name}_pope_${type}_seed${seed}_naive.jsonl \
--noise_step $noise_step \
--seed $seed  > qwen_calibrate_naive_${dataset_name}_pope_${type}_seed${seed}.out

CUDA_VISIBLE_DEVICES=0 python eval/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate/qwen/qwen_${dataset_name}_pope_${type}_seed${seed}_cd.jsonl \
--noise_step $noise_step \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_cd > qwen_calibrate_cd_${dataset_name}_pope_${type}_seed${seed}.out

CUDA_VISIBLE_DEVICES=0 python eval/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate/qwen/qwen_${dataset_name}_pope_${type}_seed${seed}_ddnone.jsonl \
--noise_step $noise_step \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd > qwen_calibrate_dd_none_${dataset_name}_pope_${type}_seed${seed}.out

CUDA_VISIBLE_DEVICES=0 python eval/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate/qwen/qwen_${dataset_name}_pope_${type}_seed${seed}_ddunk.jsonl \
--noise_step $noise_step \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd_unk > qwen_calibrate_dd_unk_${dataset_name}_pope_${type}_seed${seed}.out

CUDA_VISIBLE_DEVICES=0 python eval/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate/qwen/qwen_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step \
--seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > qwen_calibrate_dd_both_${dataset_name}_pope_${type}_seed${seed}.out