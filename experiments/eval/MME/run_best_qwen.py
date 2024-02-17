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
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel

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
    def __init__(self, questions, image_folder, tokenizer, model, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.model = model
        # self.init_image_tensors()

    def init_image_tensors(self, ):
        self.images = []
        for line in self.questions:
            image_file = line["image"]
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.model.transformer.visual.image_transform(image)
            self.images.append(image_tensor)

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        image_path = os.path.join(self.image_folder, image_file)

        question = '<img>{}</img>{} Answer:'.format(image_path, qs)
        input_ids = self.tokenizer(question, return_tensors='pt', padding='longest').input_ids

        # image_tensor = self.images[index]

        return input_ids, None, None, line

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, lines = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, lines


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, model, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, model, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

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

        with torch.inference_mode():
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
    model_name = 'qwen-vl' if 'chat' not in model_path else 'qwen-vl-chat'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    filtered_questions = [q for q in questions if q["category"] in list_subsets]
    questions = filtered_questions

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # data_loader = create_data_loader(questions, args.image_folder, tokenizer, model, model.config)

    for line in tqdm(questions, total=len(questions)):
        idx = line["question_id"]
        question = line["text"]

        image_file = line["image"]

        image_path = os.path.join(args.image_folder, image_file)
        question = '<img>{}</img>{} Answer:'.format(image_path, question)
        input_ids = tokenizer([question], return_tensors='pt', padding='longest')

        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = model.transformer.visual.image_transform(image_tensor).unsqueeze(0).to(model.device)

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None     

        with torch.inference_mode():
            model_outputs= model.generate(
                input_ids=input_ids.input_ids.cuda(),
                attention_mask=input_ids.attention_mask.cuda(),
                do_sample=True if args.temperature > 0 else False,
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
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]

        tokens_naive = calibrate_label_dict(scores, tokenizer)

        p_c_none, tokens_none, attention_none, input_ids_none = calibrate_label_sapce([line], model, tokenizer, use_tqdm=False)

        p_c_unk, tokens_unk, attention_unk, _ = calibrate_label_sapce([line], model, tokenizer, images='unk',use_tqdm=False)

        outputs = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in output_ids
        ][0]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
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
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data1/yifan/AIGC/datasets/MME_Benchmark")
    parser.add_argument("--question-file", type=str, default="./eval/MME/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="./eval/MME/answers/llava-v1.5-7b-setting.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
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
    
    args.answers_file = answers_file.replace('setting', 'default')
    eval_model(args)
    args = default_args

    if args.use_dd or args.use_dd_unk:
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