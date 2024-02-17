import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from transformers import set_seed
from utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict, get_prob_from_logits
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

import numpy as np
from PIL import Image
import math

LABEL_DICT = {0: ['yes'], 1: ['no']}
LABEL_TO_INT = {'yes': 0, 'no': 1}
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

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
            images = images.to(dtype=torch.float16, device='cuda', non_blocking=True) if images is not None else None
            model_outputs = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
                output_scores=True,
                return_dict_in_generate=True)
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]
        probs_w_token = calibrate_label_dict(scores, tokenizer, apply_softmax=True)
        all_p_y.append(get_prob_from_logits(probs_w_token))
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    if not use_tqdm:
        return p_y, probs_w_token, None, input_ids
    return p_y, None

def eval_model(args):
    # Model

    list_subsets = ["existence", "count", "position", "color", "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    filtered_questions = [q for q in questions if q["category"] in list_subsets]
    questions = filtered_questions

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None     

        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                
                images_cd=image_tensor_cd.to(dtype=torch.float16, device='cuda', non_blocking=True) if image_tensor_cd is not None else None,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
                use_dd = args.use_dd,
                use_dd_unk = args.use_dd_unk,
                output_scores=True,
                return_dict_in_generate=True
            )
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]


        tokens_naive = calibrate_label_dict(scores, tokenizer)

        image_noise = add_diffusion_noise(image_tensor, 999)
        image_zero = torch.zeros_like(image_tensor)
        image_one = torch.ones_like(image_tensor)

        p_c_none, tokens_none, attention_none, input_ids_none = calibrate_label_sapce([line], model, tokenizer, use_tqdm=False)

        p_c_unk, tokens_unk, attention_unk, _ = calibrate_label_sapce([line], model, tokenizer, images='unk',use_tqdm=False)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "naive": tokens_naive,
                                    "none": tokens_none,
                                    "unk": tokens_unk,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data1/yifan/AIGC/datasets/MME_Benchmark")
    parser.add_argument("--question-file", type=str, default="./eval/MME/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="./eval/MME/answers/llava-v1.5-7b-setting.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    args = parser.parse_args()

    import copy
    import numpy as np
    default_args = copy.deepcopy(args)
    answers_file = copy.deepcopy(args.answers_file)
    
    args.temperature = 1.0
    args.answers_file = answers_file.replace('setting', 'default')
    eval_model(args)
    args = default_args
    
    if not (args.use_dd or args.use_dd_unk):
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