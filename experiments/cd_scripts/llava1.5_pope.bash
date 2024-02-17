seed=55
dataset_name=${2:-"coco"}
type=${3:-"adversarial"}
model_path=${4:-"liuhaotian/llava-v1.5-13b"}
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

# CUDA_VISIBLE_DEVICES=4 nohup python ./eval/test_samples.py \
# --model-path ${model_path} \
# --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
# --image-folder ${image_folder} \
# --answers-file /output/llava15_${dataset_name}_pope_${type}_answers_13b_seed${seed}_cb_cut_weight${cb_cut_weight}_cb_m_weight${cb_m_weight}.jsonl \
# --noise_step $noise_step \
# --seed $seed --cb_cut_weight $cb_cut_weight --cb_m_weight $cb_m_weight > cb_cut_weight$cb_cut_weight'_'cb_m_weight$cb_m_weight.log 2>&1 &

CUDA_VISIBLE_DEVICES=6,7 nohup python ./eval/test_samples.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file /output/llava15_${dataset_name}_pope_${type}_answers_13b_seed${seed}_cb_cut_weight${cb_cut_weight}_cb_m_weight${cb_m_weight}.jsonl \
--noise_step $noise_step \
--seed $seed --cb_cut_weight $cb_cut_weight --cb_m_weight $cb_m_weight > cb_cut_weight$cb_cut_weight'_'cb_m_weight$cb_m_weight.log 2>&1 &
