import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

import kornia
import numpy as np
from lavis.models import load_model_and_preprocess
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
from utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict, get_prob_from_logits, LABEL_DICT

evolve_vcd_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def calibrate_label_sapce(questions, model, tokenizer, images=None,label_dict=LABEL_DICT, top_k=10, content_free_inputs=('N/A',), use_tqdm=True, use_image=True):
    all_p_y = []
    questions = tqdm(questions) if use_tqdm else questions
    for line in questions:
        qs = line["text"]
        prompt = qs +  " Please answer this question with one word."
        with torch.inference_mode():
            outputs, scores = model.generate({"image": images, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=None, cd_beta = args.cd_beta, use_image=use_image)
        probs_w_token = calibrate_label_dict(scores, tokenizer, apply_softmax=True)
        all_p_y.append(get_prob_from_logits(probs_w_token))
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    if not use_tqdm:
        return p_y, probs_w_token
    return p_y

def eval_model(args):
    # Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        prompt = qs +  " Please answer this question with one word."

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        ## create a white image for contrastive decoding
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        with torch.inference_mode():
            outputs, scores = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=image_tensor_cd, cd_beta = args.cd_beta)

        tokens_naive = calibrate_label_dict(scores, model.llm_tokenizer)
        p_y = get_prob_from_logits(tokens_naive)
        
        # p_c_none, tokens_none = calibrate_label_sapce([line], model, model.llm_tokenizer, images=image_tensor, use_image=False, use_tqdm=False)
        
        image_noise = add_diffusion_noise(image_tensor, 999)
        p_c_noise, tokens_noise = calibrate_label_sapce([line], model, model.llm_tokenizer, images=image_noise, use_tqdm=False)
        
        image_zero = torch.zeros_like(image_tensor)
        p_c_zero, tokens_zero = calibrate_label_sapce([line], model, model.llm_tokenizer, images=image_zero, use_tqdm=False)
        outputs = outputs[0]
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "model_id": "instruct_blip",
                                   "image": image_file,
                                   "naive": tokens_naive,
                                    "noise": tokens_noise,
                                    # "none": tokens_none,
                                    "zeros": tokens_zero,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="../data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./output/instructclip_coco_pope_adversarial_answers_seed55.jsonl")
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
