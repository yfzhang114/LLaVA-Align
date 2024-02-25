seed=55
dataset_name=${1:-"aokvqa"}
type="adversarial"
model_path=${4:-"Qwen/Qwen-VL-Chat"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}
save_dir=${2:-'default'}
temp=${3:-0.2}
# nohup bash /mnt/workspace/xue.w/yf/VCD/experiments/eval/calibrate/run_qwen.sh > pope_coco_qwen_chat_temp02_all.out 2>&1 &
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/mnt/data/xue.w/yf/data/coco/val2014
else
  image_folder=/mnt/data/xue.w/yf/data/gqa/images
fi

python eval/calibrate/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/$save_dir/qwen_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step \
--temperature $temp --seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > qwen_best_dd$save_dir${dataset_name}_pope_${type}_seed${seed}.out


type="popular"

python eval/calibrate/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/$save_dir/qwen_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step \
--temperature $temp --seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > qwen_best_dd$save_dir${dataset_name}_pope_${type}_seed${seed}.out


type="random"

python eval/calibrate/qwen_calibrate.py \
--model-path ${model_path} \
--question-file /mnt/data/xue.w/yf/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/calibrate_best_sample/$save_dir/qwen_${dataset_name}_pope_${type}_seed${seed}_both.jsonl \
--noise_step $noise_step \
--temperature $temp --seed $seed  --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd --use_dd_unk > qwen_best_dd$save_dir${dataset_name}_pope_${type}_seed${seed}.out

