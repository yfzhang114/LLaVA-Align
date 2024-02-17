import os
import json
import argparse
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict, get_prob_from_logits


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment',
                        type=str,
                        required=True)

    args = parser.parse_args()
    return args


def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT

if __name__ == "__main__":

    args = get_args()

    GT = get_gt(
        data_path='/mnt/data/xue.w/yf/data/MME_Benchmark'
    )

    experiment = args.experiment


    answers = [json.loads(line) for line in open(os.path.join('eval/MME/answers', f'{experiment}.jsonl'))]

    
    prob = {}
    labels = []
    mode = 'diagonal_W'
    num_classes =  2
    import numpy as np
    def calibrate_weight(p_cf):
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False
        return W, b
    
    LABEL_DICT = {0: 'yes', 1: 'no'}
    LABEL_TO_INT = {'yes': 0, 'no': 1}
    for name in ['naive', 'noise', 'none', 'zero', 'one']:
        prob[name] = []
    for index, line in enumerate(answers):
        for name in ['naive', 'noise', 'none', 'zero', 'one']:
            token_dict = answers[index][name]
            prob[name].append(get_prob_from_logits(token_dict))
    calibrate_mode = 'individual'
    for name in ['naive', 'noise', 'none', 'zero', 'one', 'noise_none', 'noise_zero', 'noise_none_zero', 'all']:

        results = defaultdict(list)
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])

        if calibrate_mode == 'all' and name != 'naive':
            if name == 'noise_none':
                all_p_y = np.array(prob['noise']) + np.array(prob['none'])
            elif name == 'noise_none_zero':
                all_p_y = np.array(prob['noise']) + np.array(prob['none']) + np.array(prob['zero'])
            elif name == 'noise_zero':
                all_p_y = np.array(prob['noise']) + np.array(prob['zero'])
            elif name == 'all':
                all_p_y = np.array(prob['noise']) + np.array(prob['none']) + np.array(prob['zero']) + np.array(prob['one'])
            else:
                all_p_y = prob[name]
            p_cf = np.mean(np.array(all_p_y), axis=0)
            p_cf = p_cf / np.sum(p_cf)
            W, b = calibrate_weight(p_cf)
        for i, answer in enumerate(answers):
            category = answer['question_id'].split('/')[0]
            file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
            question = answer['prompt']
            if name == 'naive':
                results[category].append((file, answer['prompt'], answer['text']))
                continue

            gen_answer = prob['naive'][i]
            if calibrate_mode == 'individual' and name != 'naive':
                if name == 'noise_none':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'noise_zero':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['zero'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'noise_none_zero':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i]) + np.array(prob['zero'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'all':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i]) + np.array(prob['zero'][i]) + np.array(prob['one'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                else:
                    p_cf = prob[name][i]
                p_cf = [x + 1e-4 for x in p_cf]
                W, b = calibrate_weight(p_cf)
            
            calibrate_label_probs = np.matmul(W, np.expand_dims(gen_answer, axis=-1)) + b
            calibrate_label_probs /= np.sum(calibrate_label_probs)

            idx = np.argmax(calibrate_label_probs)
            results[category].append((file, answer['prompt'], LABEL_DICT[idx].capitalize()))

        result_dir = os.path.join('eval/MME/eval_tool', 'answers', f'{experiment}_{name}')
        os.makedirs(result_dir, exist_ok=True)
        for category, cate_tups in results.items():
            with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
                for file, prompt, answer in cate_tups:
                    if 'Answer the question using a single word or phrase.' in prompt:
                        prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                    if 'Please answer yes or no.' not in prompt:
                        prompt = prompt + ' Please answer yes or no.'
                        if (category, file, prompt) not in GT:
                            prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
                    gt_ans = GT[category, file, prompt]
                    tup = file, prompt, gt_ans, answer
                    fp.write('\t'.join(tup) + '\n')
            fp.close()
