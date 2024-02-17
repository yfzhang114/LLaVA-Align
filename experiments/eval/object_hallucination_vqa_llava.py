import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
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

# def plt_attention_map(attention_map, input_ids, name='heatmap.jpg'):
    
#     # attention_map = attention_map.detach().cpu().numpy()
#     attention_last = attention_map[-2]
#     # ax = sns.heatmap(attention_map, cmap='viridis',  fmt='.2f', cbar=True)#annot=True,
#     ax = sns.lineplot(attention_last)
#     if input_ids is not None:
#         input_ids = input_ids[0].detach().cpu()
#         input_ids = input_ids.tolist()
#         image_token_index = input_ids.index(-200)
#         print('text atten')
#         ax.text(image_token_index + 0.5, -0.5, 'Start', ha='center', va='center', rotation=0, color='red', fontsize=8)
#         ax.text(image_token_index + 576 + 0.5, -0.5, 'End', ha='center', va='center', rotation=0, color='red', fontsize=8)
#         # ax.text(-0.5, image_token_index + 0.5, 'Start', ha='center', va='center', rotation=90, color='red', fontsize=8)
#         # ax.text(-0.5, image_token_index + 576 + 0.5, 'End', ha='center', va='center', rotation=90, color='red', fontsize=8)
    
#     plt.savefig(name)

def plt_attention_map(attention_map, input_ids, name='heatmap.jpg', tokenizer=None):
    sns.set(style="whitegrid", font_scale=1.8)
    plt.rcParams["font.family"] = "Times New Roman"
    colors = sns.color_palette("flare", n_colors=10)
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    attention_map = attention_map.detach().cpu().numpy()
    attention_last = attention_map[-2]
    ax = sns.lineplot(attention_last, color=colors[9])
    input_ids = input_ids[0].detach().cpu()
    input_ids = input_ids.tolist()
    if -200 in input_ids:
        image_token_index = input_ids.index(-200)
        ax.plot([image_token_index + 0.5, image_token_index + 576 + 0.5], [-0.05, -0.05], color=colors[1], linewidth=1.8)
        
        input_ids = input_ids[:image_token_index] + [-200 for i in range(576)] + input_ids[image_token_index+1:]
        image_token_attention_sum = attention_last[image_token_index:image_token_index + 576].sum()
        non_image_token_attention_sum = attention_last[:image_token_index].sum() + attention_last[image_token_index + 576:].sum()
        
        # 在图中添加标记
        ax.text(image_token_index + 100, 0.1, f'Image: {image_token_attention_sum:.2f}', ha='center', va='bottom', color=colors[3])
        ax.text(image_token_index + 100, 0.05, f'Text: {non_image_token_attention_sum:.2f}', ha='center', va='bottom', color=colors[2])
    else:
        image_token_index = 10240
    
    top_indices = np.argsort(attention_last)[::-1][:5]
    top_tokens = np.array(input_ids)[top_indices]
    for i, (token, prob) in enumerate(zip(top_tokens, attention_last[top_indices])):
        if top_indices[i] >= image_token_index and top_indices[i] < image_token_index + 576:
            token_string = f'Image token {top_indices[i] - 576}'
        else:
            token_string = tokenizer.batch_decode([token])
        plt.annotate(f'{token_string}\n({prob.item():.2f})',
                    xy=(top_indices[i], prob.item()),
                    xytext=(i * (len(input_ids) // 5), 0.2), rotation=45, color=colors[4])

    sns.despine(offset=10, trim=True)
    plt.ylabel('Probability')
    plt.xlabel('Token Index')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

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
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True)
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]
            attentions = model_outputs['attentions'][0][-1]
            # number_patch: 576
            attention = torch.mean(attentions, dim=1).squeeze() 
            plt_attention_map(attention, input_ids, f'{idx}_naive.pdf', tokenizer)

        tokens_naive = calibrate_label_dict(scores, tokenizer)
        p_y = get_prob_from_logits(tokens_naive)
        
        image_noise = add_diffusion_noise(image_tensor, 999)
        image_zero = torch.zeros_like(image_tensor)
        image_one = torch.ones_like(image_tensor)
        
        p_c_none, tokens_none, attention_none, input_ids_none = calibrate_label_sapce([line], model, tokenizer, use_tqdm=False)
        plt_attention_map(attention_none, input_ids_none, f'{idx}_none.pdf', tokenizer)
        acc_none, calibrate_probs_none = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_none)

        p_c_noise, tokens_noise, attention_noise, _ = calibrate_label_sapce([line], model, tokenizer, images=image_noise, use_tqdm=False)
        plt_attention_map(attention_noise, input_ids, f'{idx}_noise.pdf', tokenizer)
        acc_noise, calibrate_probs_noise = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_noise)
        
        p_c_zero, tokens_zero, attention_zero, _ = calibrate_label_sapce([line], model, tokenizer, images=image_zero, use_tqdm=False)
        acc_zero, calibrate_probs_zero = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_zero)
        
        p_c_ones, tokens_ones, attention_one, _ = calibrate_label_sapce([line], model, tokenizer, images=image_one, use_tqdm=False)
        plt_attention_map(attention_one, input_ids, f'{idx}_one.pdf', tokenizer)
        acc_one, calibrate_probs_one = eval_accuracy(np.array([p_y]), np.array([LABEL_TO_INT[line["label"]]]), p_cf=p_c_zero)
        

        all_p_y.append(p_y); all_p_c_none.append(p_c_none); all_p_c_noise.append(p_c_noise)

            
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
                                    "noise": tokens_noise,
                                    "none": tokens_none,
                                    "zero": tokens_zero,
                                    "one": tokens_ones,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./output/llava15_coco_pope_adversarial_answers.jsonl")
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
