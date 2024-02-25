cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# nohup bash ./eval/MMMU/mmmu.sh > mmmu_calibrate_7b_dd.out 2>&1 &
python ./eval/MMMU/run_qwen_sampling.py \
--output_path ./output/calibrate/llava_mmmu/qwen_naive.json \
--model_path Qwen/Qwen-VL \
--config_path ./eval/MMMU/configs/llava1.5.yaml 

python ./eval/MMMU/run_qwen_sampling.py \
--output_path ./output/calibrate/llava_mmmu/qwen_cd.json \
--model_path Qwen/Qwen-VL --cd_alpha $cd_alpha --cd_beta $cd_beta --use_cd \
--config_path ./eval/MMMU/configs/llava1.5.yaml

 python ./eval/MMMU/run_qwen_sampling.py \
--output_path ./output/calibrate/llava_mmmu/qwen_dd_both.json \
--model_path Qwen/Qwen-VL --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd_unk --use_dd \
--config_path ./eval/MMMU/configs/llava1.5.yaml


#  python ./eval/MMMU/run_qwen_sampling.py \
# --output_path ./output/calibrate/llava_mmmu/qwen_dd_none.json \
# --model_path Qwen/Qwen-VL --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd \
# --config_path ./eval/MMMU/configs/llava1.5.yaml

#  python ./eval/MMMU/run_qwen_sampling.py \
# --output_path ./output/calibrate/llava_mmmu/qwen_dd_unk.json \
# --model_path Qwen/Qwen-VL --cd_alpha $cd_alpha --cd_beta $cd_beta --use_dd_unk \
# --config_path ./eval/MMMU/configs/llava1.5.yaml



# python main_eval_calibrate.py --output_path ./output/calibrate/llava_mmmu/qwen_dd_both.json 

