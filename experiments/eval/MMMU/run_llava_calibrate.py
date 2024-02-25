import torch
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from eval.MMMU.utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from eval.MMMU.utils.model_utils import call_llava_engine_df_calibrate, llava_image_processor
from eval.MMMU.utils.eval_utils import parse_multi_choice_response, parse_open_response
from utils.metrics import ECELoss, eval_accuracy, get_prob_from_logits

from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

LABEL_DICT = None

def calibrate_label_dict(scores, tokenizer, label_dict=LABEL_DICT, top_k=100, apply_softmax=True, content_free_inputs=('N/A',)):
    special_str = '( )'
    choice_str = 'A B C D E F a b c d e f'
    special_tokens = tokenizer.encode(special_str)
    choice_token = tokenizer.encode(choice_str)

    tokens_first = [scores[i].argmax() for i in range(len(scores))]
    str_first = tokenizer.decode(tokens_first, skip_special_tokens=True)

    for i, logits in enumerate(scores):
        probs = logits.float().cpu() if not apply_softmax else torch.softmax(logits, dim=-1).float().cpu()
        top_probs, top_tokens = torch.topk(probs, k=top_k)
        
        if top_tokens[0][0] in special_tokens: #not in special_tokens and i != len(scores) - 1:# special_tokens:
            continue
        temp = {}
        for prob, token in zip(top_probs[0], top_tokens[0]):
            str_token = tokenizer.decode(token.item())
            str_token = str_token.lower().strip()
            if str_token not in temp.keys():
                temp[str_token] = prob.item()
            else:
                pass
        # if list(temp.keys())[0] in choice_str.split():
        #     return temp
        return temp
    return temp

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()

    def calibrate_weight(p_cf, mode='diagonal_W', num_classes=4):
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False
        return W, b
    
    prob = dict()
    for name in ['naive', 'noise', 'none', 'zero', 'one', 'unk']:
        prob[name] = []
    naive_response, responses = [], []
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)
            naive_response.append(response['response'])
            responses.append(response)

            if 'all_choices' not in sample.keys():
                for name_ in ['naive', 'noise', 'none', 'zero', 'one', 'unk']:
                    prob[name_].append([])
                continue
            global LABEL_DICT
            for i, choice in enumerate(sample['all_choices']):
                LABEL_DICT[i] = choice
            for name_ in ['naive', 'noise', 'none', 'zero', 'one', 'unk']:
                prob[name_].append(get_prob_from_logits(response[name_], label_dict=LABEL_DICT))

    calibrate = 'all'
    for name in ['naive', 'noise', 'none', 'none_pure', 'zero', 'unk', 'unk_pure', 'none_noise', 'none_unk', 'none_unk_noise', 'all']:
        i = 0
        out_samples[name] = dict()
        for sample in tqdm(samples):
            if 'all_choices' in sample.keys():
                num_classes = len(sample['all_choices'])
                gen_answer = prob['naive'][i]
                if 'pure' in name:
                    gen_answer = prob[name[:-5]][i]
                gen_answer /= np.sum(gen_answer)
                if name == 'none_noise':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'none_unk':
                    all_p_y = np.array(prob['unk'][i]) + np.array(prob['zero'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'none_unk_noise':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i]) + np.array(prob['unk'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'all':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i]) + np.array(prob['zero'][i]) + np.array(prob['unk'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif 'pure' in name:
                    p_cf = np.array(prob[name[:-5]][i])
                else:
                    p_cf = np.array(prob[name][i])
                    # p_cf = p_cf / np.sum(p_cf)

                p_cf = [x + 1e-4 for x in p_cf]
                if name == 'naive' or 'pure' in name or min(p_cf) <= 1e-4:
                    W = np.identity(num_classes)
                    b = np.zeros([num_classes, 1])
                else:
                    W, b = calibrate_weight(p_cf, num_classes=num_classes)

                calibrate_label_probs = np.matmul(W, np.expand_dims(gen_answer, axis=-1)) + b
                calibrate_label_probs /= np.sum(calibrate_label_probs)

                idx = np.argmax(calibrate_label_probs)
                response = sample['all_choices'][idx]
            else:
                response = naive_response[i]
            ### calibrate the probability 
            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[name][sample['id']] = pred_ans
            i += 1
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calibrate_label_sapce(questions, model, tokenizer, images=None, images_custom=None, label_dict=None, content_free_inputs=('N/A',), use_tqdm=True):
    all_p_y = []
    questions = tqdm(questions) if use_tqdm else questions
    for prompt in questions:

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if images == 'unk':
            input_ids[input_ids == IMAGE_TOKEN_INDEX] = tokenizer.unk_token_id
            images = None
        elif images == None:
            input_ids = input_ids[input_ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
        with torch.inference_mode():
            images = images.cuda() if images is not None else None
            model_outputs = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                # num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=128,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True)
            output_ids = model_outputs['sequences']
        probs_w_token = calibrate_label_dict(model_outputs['scores'], tokenizer, apply_softmax=True, label_dict=LABEL_DICT)
        all_p_y.append(get_prob_from_logits(probs_w_token, label_dict=LABEL_DICT))
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    if not use_tqdm:
        return p_y, probs_w_token
    return p_y

def call_llava_engine_df_calibrate(args, sample, model, tokenizer=None, processor=None):
    
    global LABEL_DICT
    LABEL_DICT = {}
    if 'all_choices' in sample.keys():
        for i, choice in enumerate(sample['all_choices']):
            LABEL_DICT[i] = choice
    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    images = sample['image'].unsqueeze(0).half().cuda() if sample['image'] is not None else sample['image']

    if args.use_cd:
        images_cd = add_diffusion_noise(images, args.noise_step)
    else:
        images_cd = None   

    model_outputs = model.generate(
                input_ids,
                images=images if images is not None else images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                # num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=128,
                images_cd=images_cd,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
                use_dd = args.use_dd,
                use_dd_unk = args.use_dd_unk,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True)
    output_ids = model_outputs['sequences']
    scores = model_outputs['scores'][0]

    tokens_naive = calibrate_label_dict(model_outputs['scores'], tokenizer, label_dict=LABEL_DICT)
    p_y = get_prob_from_logits(tokens_naive, label_dict=LABEL_DICT)
    
    image_noise = add_diffusion_noise(images, 999)
    image_zero = torch.zeros_like(images)
    image_one = torch.ones_like(images)

    if 'all_choices' in sample.keys():
        p_c_none, tokens_none = calibrate_label_sapce([prompt.replace('<image>', '')], model, tokenizer, images=None, use_tqdm=False)

        p_c_unk, tokens_unk = calibrate_label_sapce([prompt.replace('<image>', '<unk/>')], model, tokenizer, images='unk',use_tqdm=False)

        p_c_noise, tokens_noise = calibrate_label_sapce([prompt], model, tokenizer, images=image_noise, use_tqdm=False)
        
        p_c_zero, tokens_zero = calibrate_label_sapce([prompt], model, tokenizer, images=image_zero, use_tqdm=False)
        
        p_c_one, tokens_one = calibrate_label_sapce([prompt], model, tokenizer, images=image_one, use_tqdm=False)
    else:
        tokens_none, tokens_unk, tokens_noise, tokens_zero, tokens_one = None, None, None, None, None

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    return {"question_id": -1,
            "prompt": prompt,
            "response": response,
            "naive": tokens_naive,
            "noise": tokens_noise,
            "none": tokens_none,
            "zero": tokens_zero,
            "unk": tokens_unk,
            "one": tokens_one,
            "model_id": model_name,
            "metadata": {}}

def llava_image_processor(raw_image, vis_processors=None):
    images = vis_processors.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    return images

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_7b_mmmu_calibrate_pure_skipzero.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="/mnt/workspace/xue.w/yf/VCD/experiments/eval/MMMU/configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--split', type=str, default='validation')

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df_calibrate
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
                                                                model_name)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    save_json(args.output_path, out_samples)