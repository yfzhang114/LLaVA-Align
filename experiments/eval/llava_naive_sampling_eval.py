import argparse
import torch
import os
import json
from tqdm import tqdm
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import numpy as np

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()


def eval_model(args):
    gt_files = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    # open generated answers
    gen_files = [json.loads(q) for q in open(os.path.expanduser(args.answer_file), "r")]

    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(gt_files)
    yes_answers = 0

    # compare answers
    for index, line in enumerate(gt_files):
        idx = line["question_id"]
        gt_answer = line["label"]
        assert idx == gen_files[index]["question_id"]
        gen_answer = gen_files[index]["text"]
        # convert to lowercase
        gt_answer = gt_answer.lower()
        gen_answer = gen_answer.lower()
        # strip
        gt_answer = gt_answer.strip()
        gen_answer = gen_answer.strip()
        # pos = 'yes', neg = 'no'
        if gt_answer == 'yes':
            if 'yes' in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == 'no':
            if 'no' in gen_answer:
                true_neg += 1
            else:
                yes_answers += 1
                false_pos += 1
        else:
            print(f'Warning: unknown gt_answer: {gt_answer}')
            unknown += 1
    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions
    # report results
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Yes': yes_proportion,
        'F1': f1,
        'Unknown': unknown_prop
    }
    return metrics_dict


def iter_over_seed(args, name):
    metrics_dicts = []
    for seed in [53, 54, 55]:
        args.answer_file = args.answer_file.replace('vs', f'{seed}')
        metrics_dicts.append(eval_model(args))
        args.answer_file = args.answer_file.replace(f'{seed}', 'vs')
    
    mean_std_dict = {'Name': name}
    for key in metrics_dicts[0].keys():
        values = [metrics[key] for metrics in metrics_dicts]
        mean_value = np.mean(values)
        std_value = np.std(values)
        # mean_std_dict[key] = f'{mean_value*100:.2f} ± {std_value:.2f}'
        mean_std_dict[key] = mean_value * 100
    return mean_std_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/mnt/data/xue.w/yf/VCD/experiments/data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answer-file", type=str, default="llava15_7b_coco_pope_adversarial_seed53_setting.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    question_file = "/mnt/data/xue.w/yf/VCD/experiments/data/POPE/coco/coco_pope_type.json"
    import pandas as pd
    types = ['adversarial', 'popular', 'random']
    results_types = []
    for type_ in types:
        results = []
        args.question_file = question_file.replace('type', type_)
        args.answer_file = f'/mnt/data/xue.w/yf/VCD/experiments/output/sampling/llava-v1.5-13b/llava15_13b_coco_pope_{type_}_seedvs_setting.jsonl'
        answer_file = copy.deepcopy(args.answer_file)
        default_args = copy.deepcopy(args)
        
        args.answer_file = answer_file.replace('setting', 'greedy')
        results.append(iter_over_seed(args, 'greedy'))
        args = default_args
            
        for temp in np.arange(0.05, 1.05, 0.05):
            temp = np.round(temp, 2)
            print(f"Running temp = {temp}")
            args.answer_file = answer_file.replace('setting', f'temp_{temp}')
            
            results.append(iter_over_seed(args, f'temp_{temp}'))
        args = default_args
        for top_p in np.arange(0, 1.05, 0.05):
            top_p = np.round(top_p, 2)
            print(f"Running top_p = {top_p}")
            args.answer_file = answer_file.replace('setting', f'top_p_{top_p}')
            results.append(iter_over_seed(args, f'top_p_{top_p}'))
        args = default_args
        
        for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
            print(f"Running top_k = {top_k}")
            args.answer_file = answer_file.replace('setting', f'top_k_{top_k}')
            results.append(iter_over_seed(args, f'top_k_{top_k}'))
        args = default_args
        results_types.append(results)
    
    df1 = pd.DataFrame(results_types[0])
    df2 = pd.DataFrame(results_types[1])
    df3 = pd.DataFrame(results_types[2])

    # 创建一个Excel写入对象
    with pd.ExcelWriter('sampling_results_7bsft.xlsx', engine='xlsxwriter') as writer:
        # 将每个DataFrame写入Excel文件的不同sheet
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        df3.to_excel(writer, sheet_name='Sheet3', index=False)