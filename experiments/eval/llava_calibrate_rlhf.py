import argparse
import torch
import os
import json
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import *

from PIL import Image
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict, get_prob_from_logits

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

# LABEL_DICT = {0: ['Yes', 'yes'], 1: ['No', 'no']}
LABEL_DICT = {0: ['yes'], 1: ['no']}
LABEL_TO_INT = {'yes': 0, 'no': 1}

def calibrate_label_sapce(questions, model, tokenizer, images=None,label_dict=LABEL_DICT, top_k=100, content_free_inputs=('N/A',), use_tqdm=True):
    all_p_y = []
    questions = tqdm(questions) if use_tqdm else questions
    for line in questions:
        qs = line["text"]
        if images is not None:
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if images == 'unk':
            input_ids[input_ids == IMAGE_TOKEN_INDEX] = tokenizer.unk_token_id
            images = None
        with torch.inference_mode():
            images = images.unsqueeze(0).half().cuda() if images is not None else None
            model_outputs = model.generate(
                input_ids,
                images=images,
                images_cd=None,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True)
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]
            attentions = model_outputs['attentions'][0][-1]
            attention = torch.mean(attentions, dim=1).squeeze() # average over the head dimension
        probs_w_token = calibrate_label_dict(scores, tokenizer, apply_softmax=True)
        all_p_y.append(get_prob_from_logits(probs_w_token))
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    if not use_tqdm:
        return p_y, probs_w_token, attention, input_ids
    return p_y, attention

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    compute_dtype = torch.float16

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=compute_dtype)
    image_processor = vision_tower.image_processor

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    
    all_p_y, all_p_c_none, all_p_c_noise, labels = [], [], [], []
    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        labels.append(LABEL_TO_INT[line["label"]])
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                use_dd = args.use_dd,
                use_dd_unk = args.use_dd_unk,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=64,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True)
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]
            attentions = model_outputs['attentions'][0][-1]
            # number_patch: 576
            attention = torch.mean(attentions, dim=1).squeeze() 
            # plt_attention_map(attention, input_ids, f'{idx}_naive.pdf', tokenizer)

        tokens_naive = calibrate_label_dict(scores, tokenizer)
        p_y = get_prob_from_logits(tokens_naive)
        
        image_noise = add_diffusion_noise(image_tensor, 999)
        image_zero = torch.zeros_like(image_tensor)
        image_one = torch.ones_like(image_tensor)
        
        p_c_none, tokens_none, attention_none, input_ids_none = calibrate_label_sapce([line], model, tokenizer, use_tqdm=False)
        
        p_c_unk, tokens_unk, attention_unk, _ = calibrate_label_sapce([line], model, tokenizer, images='unk',use_tqdm=False)

        all_p_y.append(p_y); all_p_c_none.append(p_c_none)
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = outputs[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "logits_score": p_y,
                                   "naive": tokens_naive,
                                    "unk": tokens_unk,
                                    "none": tokens_none,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data3/yifanzhang/.cache/huggingface/hub/LLaVA-RLHF-7b-v1.5-224/sft_model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./output/llava15_coco_pope_adversarial_answers_t02_beam1.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use-qlora", type=bool, default=True)
    parser.add_argument("--qlora-path", type=str, default="/data3/yifanzhang/.cache/huggingface/hub/LLaVA-RLHF-7b-v1.5-224/rlhf_lora_adapter_model")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    print(args)
    eval_model(args)
