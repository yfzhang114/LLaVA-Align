import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import numpy as np

from utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

# LABEL_DICT = {0: ['Yes', 'yes'], 1: ['No', 'no']}
LABEL_DICT = {0: ['yes'], 1: ['no']}
LABEL_TO_INT = {'yes': 0, 'no': 1}


def get_prob_from_logits(top_token_probs):
    return [0, 0]

def calibrate_label_sapce(questions, model, tokenizer, images=None,label_dict=LABEL_DICT, top_k=100, content_free_inputs=('N/A',), use_tqdm=True):
    all_p_y, tokens_w_probs = [], []
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
                return_dict_in_generate=True)
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]
        probs_w_token = calibrate_label_dict(scores, tokenizer, apply_softmax=True)
        tokens_w_probs.append(probs_w_token)
        all_p_y.append(get_prob_from_logits(probs_w_token))
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y, tokens_w_probs[0]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for question_name in ['actions','numbers', 'colors', 'relations', 'shapes' ]:
        question_file = f'data/POPE/coco/all_coco_{question_name}.json'
        answers_file = f'output/all_coco_{question_name}.json'
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        image_tensor = None
        
        all_p_y, all_p_c_none, all_p_c_noise, labels = [], [], [], []
        
        idx = 0
        for line in tqdm(questions):
            idx += 1
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

            if image_tensor is None:
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            if args.use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                image_tensor_cd = None      

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            p_y = [0, 0]
            p_c_none, tokens_none = calibrate_label_sapce([line], model, tokenizer, use_tqdm=False)
            acc_none, calibrate_probs_none = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_none)
            
            image_noise = add_diffusion_noise(image_tensor, 999)
            p_c_noise, tokens_noise = calibrate_label_sapce([line], model, tokenizer, images=image_noise, use_tqdm=False)
            acc_noise, calibrate_probs_noise = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_noise)
            
            image_zero = torch.zeros_like(image_tensor)
            p_c_zero, tokens_zero = calibrate_label_sapce([line], model, tokenizer, images=image_zero, use_tqdm=False)
            acc_noise, calibrate_probs_noise = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_noise)
            
            image_ones = torch.ones_like(image_tensor)
            p_c_one, tokens_ones = calibrate_label_sapce([line], model, tokenizer, images=image_ones, use_tqdm=False)
            acc_one, calibrate_probs_one = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_noise)
            
            all_p_y.append(p_y); all_p_c_none.append(p_c_none); all_p_c_noise.append(p_c_noise)

            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "model_id": model_name,
                                    "image": image_file,
                                    "logits_score": p_y,
                                    "noise": tokens_noise,
                                    "none": tokens_none,
                                    "zeros": tokens_zero,
                                    "ones": tokens_ones,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./data/POPE/coco/test_samples_new.json")
    parser.add_argument("--answers-file", type=str, default="./output/test_samples.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--calibrate", action='store_true', default=False)
    parser.add_argument("--calibrate_each", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
