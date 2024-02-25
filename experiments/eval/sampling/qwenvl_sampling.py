import argparse
import torch
import os
import json
from tqdm import tqdm
import copy
import sys
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

import kornia
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen-vl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    print(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    if 'POPE' not in args.question_file:
        max_new_tokens = 1024
    else:
        max_new_tokens = 20
    for line in tqdm(questions, miniters=100):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        question = 'Question: <img>{}</img> {} Answer:'.format(image_path, question)
        questions_id = []
        input_ids = tokenizer([question], return_tensors='pt', padding='longest')

        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = model.transformer.visual.image_transform(image_tensor).unsqueeze(0).to(model.device)

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None   
        if 'POPE' in args.question_file:
            pred = model.generate(
                input_ids=input_ids.input_ids.cuda(),
                attention_mask=input_ids.attention_mask.cuda(),
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_dd = args.use_dd,
                use_dd_unk = args.use_dd_unk,
                images = image_tensor,
                images_cd=image_tensor_cd,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
            )
        else:
            model.generation_config.max_new_tokens=max_new_tokens
            pred = model.generate(
                input_ids=input_ids.input_ids.cuda(),
                attention_mask=input_ids.attention_mask.cuda(),
                do_sample=True if args.temperature > 0 else False,
                use_cache=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_dd = args.use_dd,
                use_dd_unk = args.use_dd_unk,
                images = image_tensor,
                images_cd=image_tensor_cd,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
            )

        outputs = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                            skip_special_tokens=True).strip() for _ in pred
        ][0]

        outputs = outputs.strip()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": question,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/data/xue.w/yf/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="./data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./output/qwen_coco_pope_adversarial_setting.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    answers_file = copy.deepcopy(args.answers_file)
    
    args.temperature = 1.0
    args.top_p = None
    args.top_k = None
    args.do_sample = True
    default_args = copy.deepcopy(args)
    print('default temp=1.0')
    args.answers_file = answers_file.replace('setting', 'default')
    eval_model(args)
    args = default_args

    if args.use_cd:
        exit()
        
    for temp in np.arange(0.05, 1.05, 0.05):
        temp = np.round(temp, 2)
        print(f"Running temp = {temp}")
        
        args.do_sample = True
        args.temperature = temp
        args.answers_file = answers_file.replace('setting', f'temp_{temp}')
        
        eval_model(args)
    args = default_args
            
    for top_p in np.arange(0, 1.05, 0.05):
        top_p = np.round(top_p, 2)
        print(f"Running top_p = {top_p}")
        
        args.do_sample = True
        args.top_p=top_p
        args.answers_file = answers_file.replace('setting', f'top_p_{top_p}')
        
        eval_model(args)
    args = default_args
    
    for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        print(f"Running top_k = {top_k}")
        args.do_sample = True
        args.top_k = top_k
        args.answers_file = answers_file.replace('setting', f'top_k_{top_k}')
        
        eval_model(args)
    args = default_args