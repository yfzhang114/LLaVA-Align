import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import sys
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import ECELoss
parser = argparse.ArgumentParser()
parser.add_argument("--gt_files", type=str, default="./experiments/data/POPE/coco/coco_pope_adversarial.json")
parser.add_argument("--gen_files", type=str, default="output/output_hav/wo_cd/llava15_coco_pope_adversarial_answers_no_cd_seed55_cb_alpha0.5.jsonl")
args = parser.parse_args()

LABEL_DICT = {0: ['yes'], 1: ['no']}
LABEL_TO_INT = {'yes': 0, 'no': 1}

def get_prob_from_logits(top_token_probs):
    top_token_probs = {key.lower().strip(): value for key, value in top_token_probs.items()}
    p_y = [0] * len(LABEL_DICT)
    for i, answers in LABEL_DICT.items():
        prob = 0
        for a in answers:
            if a not in top_token_probs.keys():
                prob += 0
            else:
                prob += top_token_probs[a]
        p_y[i] = prob
    return p_y

confidence_low=0.
confidence_high=1.
seed=55
data = 'gqa' # coco # gqa #aokvqa
# for seed in [55]: #53, 54, 
for split in ['random', 'popular', 'adversarial', ]:
    print('*'*20+f'split {split}' + '*'*20)
    args.gt_files = f'./experiments/data/POPE/{data}/{data}_pope_{split}.json'
    args.gen_files = f'./experiments/output/llava-13B/llava_{data}_pope_{split}_seed55_both.jsonl'
    print(args.gen_files)

    # open ground truth answers
    gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]

    # open generated answers
    gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]

    # compare answers
    prob = {}
    labels = []
    for name in ['naive', 'noise', 'none', 'zero', 'unk']:
        prob[name] = []
    for index, line in enumerate(gt_files):
        idx = line["question_id"]
        gt_answer = line["label"]
        assert idx == gen_files[index]["question_id"]
        
        labels.append(LABEL_TO_INT[gt_answer])
        for name in ['naive', 'noise', 'none', 'zero', 'unk']:
            if name not in gen_files[index].keys():
                continue
            token_dict = gen_files[index][name]
            prob[name].append(get_prob_from_logits(token_dict))

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
    mode = 'diagonal_W'
    num_classes =  2
    scores_naive = prob['naive']
    calibrate_mode = 'individual'
    ece_loss = ECELoss(10)

    ece_naive = ece_loss(scores_naive, labels)
    # for name in ['naive', 'noise', 'none', 'zero', 'unk', 'none_noise', 'none_unk', 'none_unk_noise', 'all']:
    for name in ['naive', 'none', 'unk', 'none_unk']:
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        unknown = 0
        yes_answers = 0
        total_questions = 0
        confidence = 0
        print(f'Evaluate the performance in {name} setting')
        calibrate_probs = []

        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
        
        if calibrate_mode == 'all' and name != 'naive':
            if name == 'none_noise':
                all_p_y = np.array(prob['noise']) + np.array(prob['none'])
            elif name == 'none_unk':
                all_p_y = np.array(prob['unk']) + np.array(prob['none'])
            elif name == 'none_unk_noise':
                all_p_y = np.array(prob['noise']) + np.array(prob['none']) + np.array(prob['unk'])
            elif name == 'noise_zero':
                all_p_y = np.array(prob['noise']) + np.array(prob['zero'])
            elif name == 'all':
                all_p_y = np.array(prob['noise']) + np.array(prob['none']) + np.array(prob['zero']) + np.array(prob['unk'])
            else:
                all_p_y = prob[name]
            p_cf = np.mean(np.array(all_p_y), axis=0)
            p_cf = p_cf / np.sum(p_cf)
            W, b = calibrate_weight(p_cf)            
        
        for i in range(len(labels)):
            gen_answer = scores_naive[i]
            gen_answer = gen_answer / np.sum(gen_answer)
            if np.max(gen_answer) > confidence_high or np.max(gen_answer) < confidence_low: 
                continue
            if calibrate_mode == 'individual' and name != 'naive':
                if name == 'none_noise':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'none_unk':
                    all_p_y = np.array(prob['unk'][i]) + np.array(prob['none'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'none_unk_noise':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i]) + np.array(prob['unk'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                elif name == 'all':
                    all_p_y = np.array(prob['noise'][i]) + np.array(prob['none'][i]) + np.array(prob['zero'][i]) + np.array(prob['unk'][i])
                    p_cf = all_p_y / np.sum(all_p_y)
                else:
                    p_cf = prob[name][i]
                    p_cf = p_cf / np.sum(p_cf)
                p_cf = [x + 1e-4 for x in p_cf]
                W, b = calibrate_weight(p_cf)
            
            calibrate_label_probs = np.matmul(W, np.expand_dims(gen_answer, axis=-1)) + b
            calibrate_label_probs /= np.sum(calibrate_label_probs)
            calibrate_probs.append(calibrate_label_probs)
            
            # convert to lowercase
            gt_answer = labels[i]
            gen_answer = np.argmax(calibrate_label_probs)
            confidence += np.max(calibrate_label_probs)
            # strip
            # pos = 'yes', neg = 'no'
            if gt_answer == 0:
                if 0 == gen_answer:
                    true_pos += 1
                    yes_answers += 1
                else:
                    false_neg += 1
            elif gt_answer == 1:
                if 1 == gen_answer:
                    true_neg += 1
                else:
                    yes_answers += 1
                    false_pos += 1
            else:
                print(f'Warning: unknown gt_answer: {gt_answer}')
                unknown += 1
            total_questions += 1
        # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / total_questions
        yes_proportion = yes_answers / total_questions
        unknown_prop = unknown / total_questions
        confidence /= total_questions
        # ece = ece_loss(calibrate_probs, labels)
        # report results
        print(f'F1: {f1*100:.4} Accuracy: {accuracy*100:.4} Precision: {precision*100:.4} \t Recall: {recall*100:.4} \t yes: {yes_proportion*100:.4} unknow: {unknown_prop*100:.4} number questions {total_questions} confidence {confidence}')