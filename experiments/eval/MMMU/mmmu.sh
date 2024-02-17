cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# nohup bash /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/mmmu.sh > mmmu_calibrate_7b_dd.out 2>&1 &

# python /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/run_llava_calibrate.py \
# --output_path /mnt/workspace/xue.w/yf/VCD/experiments/output/calibrate/llava_mmmu/llava1.5_7b_naive.json \
# --model_path liuhaotian/llava-v1.5-7b \
# --config_path /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/configs/llava1.5.yaml 

python /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/run_llava_calibrate.py \
--output_path /mnt/workspace/xue.w/yf/VCD/experiments/output/calibrate/llava_mmmu/llava1.5_7b_cd.json \
--model_path liuhaotian/llava-v1.5-7b --cd_alpha $cd_alpha --cd_beta $cd_beta --use_cd \
--config_path /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/configs/llava1.5.yaml

 python /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/run_llava_calibrate.py \
--output_path /mnt/workspace/xue.w/yf/VCD/experiments/output/calibrate/llava_mmmu/llava1.5_7b_dd_both.json \
--model_path liuhaotian/llava-v1.5-7b --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd_unk --use_dd \
--config_path /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/configs/llava1.5.yaml


 python /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/run_llava_calibrate.py \
--output_path /mnt/workspace/xue.w/yf/VCD/experiments/output/calibrate/llava_mmmu/llava1.5_7b_dd_none.json \
--model_path liuhaotian/llava-v1.5-7b --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd \
--config_path /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/configs/llava1.5.yaml

#  python /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/run_llava_calibrate.py \
# --output_path /mnt/workspace/xue.w/yf/VCD/experiments/output/calibrate/llava_mmmu/llava1.5_7b_dd_unk.json \
# --model_path liuhaotian/llava-v1.5-7b --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd_unk \
# --config_path /mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/configs/llava1.5.yaml



# python main_eval_calibrate.py --output_path /mnt/workspace/xue.w/yf/VCD/experiments/output/calibrate/llava_mmmu/llava1.5_7b_dd_both.json 

