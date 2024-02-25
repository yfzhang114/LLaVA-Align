seed=53
size=7
model_path=${4:-"liuhaotian/llava-v1.5-13b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}

image_folder=/mnt/data/xue.w/yf/data/coco/val2014/
# nohup bash /mnt/workspace/xue.w/yf/VCD/experiments/cd_scripts/llava_bench.bash >> llava_bench_13B.log 2>&1 &

python ./eval/sampling/llava_sampling.py \
--model-path ${model_path} \
--question-file data/qa90_questions.jsonl \
--image-folder ${image_folder} \
--answers-file ./output/llava_bench/13B/llava_bench_naive_seed${seed}_setting.jsonl \
--noise_step $noise_step \
--seed $seed

python ./eval/sampling/llava_sampling.py \
--model-path ${model_path} \
--question-file data/qa90_questions.jsonl \
--image-folder ${image_folder} \
--answers-file ./output/llava_bench/13B/llava_bench_vcd_seed${seed}_setting.jsonl \
--noise_step $noise_step \
--seed $seed --use_cd

python ./eval/sampling/llava_sampling.py \
--model-path ${model_path} \
--question-file data/qa90_questions.jsonl \
--image-folder ${image_folder} \
--answers-file ./output/llava_bench/13B/llava_bench_vdd_seed${seed}_setting.jsonl \
--noise_step $noise_step \
--seed $seed --use_dd_unk --use_dd
