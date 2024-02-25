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
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
from utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict, get_prob_from_logits, LABEL_DICT
import numpy as np
import kornia
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel

from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()


def calibrate_label_sapce(questions, model, tokenizer, images=None,label_dict=LABEL_DICT, top_k=100, content_free_inputs=('N/A',), use_tqdm=True):
    all_p_y = []
    questions = tqdm(questions) if use_tqdm else questions
    for line in questions:
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        if images is None:
            question = '{} Answer:'.format(question)
        elif images == 'unk':
            question = '{} {} Answer:'.format('None', question)
        else:
            question = '<img>{}</img>{} Answer:'.format(image_path, question)
        input_ids = tokenizer([question], return_tensors='pt', padding='longest')
        
        model_outputs= model.generate(
            input_ids=input_ids.input_ids.cuda(),
            attention_mask=input_ids.attention_mask.cuda(),
            do_sample=True,
            max_new_tokens=20,
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
            images = images,
            images_cd=None,
            cd_beta = args.cd_beta,
            cd_alpha = args.cd_alpha,
            output_scores=True,
            return_dict_in_generate=True
        )
        pred = model_outputs['sequences']
        scores = model_outputs['scores'][0]
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
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen-vl' if 'Chat' not in model_path else 'qwen-vl-chat'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True).eval()


    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        question = '<img>{}</img>{} Answer:'.format(image_path, question)
        questions_id = []
        input_ids = tokenizer([question], return_tensors='pt', padding='longest')

        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = model.transformer.visual.image_transform(image_tensor).unsqueeze(0).to(model.device)
    
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None   
        
        model_outputs= model.generate(
            input_ids=input_ids.input_ids.cuda(),
            attention_mask=input_ids.attention_mask.cuda(),
            do_sample=True,
            max_new_tokens=20,
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
            images = image_tensor,
            images_cd=image_tensor_cd,
            cd_beta = args.cd_beta,
            cd_alpha = args.cd_alpha,
            use_dd = args.use_dd,
            use_dd_unk = args.use_dd_unk,
            output_scores=True,
            return_dict_in_generate=True
        )
        pred = model_outputs['sequences']
        scores = model_outputs['scores'][0]

        tokens_naive = calibrate_label_dict(scores, tokenizer)
        p_y = get_prob_from_logits(tokens_naive)
        
        p_c_none, tokens_none = calibrate_label_sapce([line], model, tokenizer, use_tqdm=False)
        p_c_unk, tokens_unk = calibrate_label_sapce([line], model, tokenizer, images='unk',use_tqdm=False)
        
        image_noise = add_diffusion_noise(image_tensor, 999)
        p_c_noise, tokens_noise = calibrate_label_sapce([line], model, tokenizer, images=image_noise, use_tqdm=False)
        
        image_zero = torch.zeros_like(image_tensor)
        p_c_zero, tokens_zero = calibrate_label_sapce([line], model, tokenizer, images=image_zero, use_tqdm=False)
                
        outputs = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ][0]
        outputs = outputs.strip()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": question,
                                   "text": outputs,
                                    "naive": tokens_naive,
                                    "noise": tokens_noise,
                                    "none": tokens_none,
                                    "zero": tokens_zero,
                                    "unk": tokens_unk,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./output/qwenvl_coco_pope_adversarial_answers_no_cd_seed55.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
