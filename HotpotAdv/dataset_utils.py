import sys

import string
import re
from collections import Counter
import numpy as np
import json

from utils import *

_ALT_ANS_FILE = 'annotations/alt_answers.jsonlines'
_ENABLING_ALT_ANSWERS = True

_INCORRECT_ANS_FILE = 'annotations/incorrect_answers.jsonlines'

def read_alternative_answers(filename=_ALT_ANS_FILE):
    with open(filename) as f:
        lines = f.readlines()
    alt_ans_dict = {}
    for l in lines:
        d = json.loads(l)
        if d['qas_id'] not in alt_ans_dict:
            alt_ans_dict[d['qas_id']] = []
        if d['answer'] not in alt_ans_dict[d['qas_id']]:
            alt_ans_dict[d['qas_id']].append(d['answer'])

    return alt_ans_dict

def read_incorrect_answers(filename=_INCORRECT_ANS_FILE):
    return read_alternative_answers(filename=filename)

def read_manual_rationale(filename):
    with open(filename) as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        id, rat = l.rstrip().split('\t')
        d[id] = rat
    return d
    
def read_hotpot_data(fname, n_dist, par_connection=' ', manual_annotation_style=None):
    data = read_json(fname)
    alt_ans_dict = read_alternative_answers()
    if manual_annotation_style:
        man_rat_dict = read_manual_rationale(f'annotations/manual_rat_{manual_annotation_style}.txt')
        man_answer_dict = read_manual_rationale(f'annotations/manual_answer.txt')
    else:
        man_rat_dict = dict()
    if manual_annotation_style is not None:
        data = [x for x in data if x["qas_id"] in man_rat_dict] # select the data with manual rationales
    examples = []
    for d in data:

        paragraphs = d["paragraphs"]
        kept_pars = []
        num_dist_added = 0
        for p in paragraphs:
            if p['is_supp']:
                kept_pars.append(p) #append supporting paragraphs
            else:
                if num_dist_added < n_dist: #append distractors
                    kept_pars.append(p)
                    num_dist_added += 1
        
        context = par_connection.join([x['context'] for x in kept_pars]) #concat all sentences as context
        
        supporting_paragraphs = []
        for i, p in enumerate(kept_pars):
            if p['is_supp']:
                supporting_paragraphs.append(i + 1) #indices of supporting paragraphs in context

        possible_answers = [d["answer"]]
        if d["qas_id"] in alt_ans_dict and _ENABLING_ALT_ANSWERS:
            possible_answers.extend(alt_ans_dict[d["qas_id"]]) #all possible answers (counted as correct if a paraphrase)
            
        ex = {
            "id": d["qas_id"],
            "question": d["question"].lstrip(),
            "answer": d["answer"],
            "context": context, #replaced with new context: supporting + distractors
            "rationale": [x['sentence'].lstrip() for x in d["rationale"]],
            "pars": [x['context'] for x in kept_pars], #replaced with new context: supporting + distractors
            "supp_pars": supporting_paragraphs, #indices of supporting paragraphs in context
            "answer_choices": possible_answers
        }
        if d["qas_id"] in man_rat_dict:            
            ex["manual_rationale"] = man_rat_dict[d["qas_id"]] #record the manual rationale if exists
            ex["manual_answer"] = man_answer_dict[d["qas_id"]]
        examples.append(ex)    
    return examples


# hotpot evaluation

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def _exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def hotpot_evaluation(prediction, ground_truth):
    return _exact_match_score(prediction, ground_truth), _f1_score(prediction, ground_truth)

def hotpot_evaluation_with_multi_answers(prediction, answers):
    best_acc, best_fpr, best_ans = False, (-1.0,.0,.0), answers[0]
    for ans in answers:
        ex, fpr = hotpot_evaluation(prediction, ans)
        if fpr[0] > best_fpr[0]: #fpr[0]: f1 score better
            best_acc, best_fpr, best_ans = ex, fpr, ans
    return best_acc, best_fpr, best_ans    

def f1auc_score(score, f1):
    score = np.array(score)
    f1 = np.array(f1)
    
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    segment = min(1000, score.size - 1)
    T = np.arange(segment) + 1
    T = T/segment
    results = np.array([np.mean(f1[:int(num_test * t)])  for t in T])
    for t in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
        print("%.1f"%(np.mean(f1[:int(num_test * t)]) * 100), end=",")
    return np.mean(results) * 100

def f1auc_curve(score, f1):
    score = np.array(score)
    f1 = np.array(f1)

    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size        
    # print(results)
    tick = 20
    x_axis = (np.arange(tick) + 1) / tick
    y = np.array([np.mean(f1[:int(num_test * t)]) for t in x_axis])    
    return x_axis, y

def train_max_accuracy(x, y):
    x = x.flatten()
    best_acc = 0
    best_v = 0
    for v in x:
        p = x > v
        ac = np.sum(p == y) / y.size
        if ac > best_acc:
            best_acc = ac
            best_v= v
    return best_acc, best_v

def merge_predication_chunks(file1, file2):
    print(file1)
    print(file2)

    file1_args = file1.split("_")
    file2_args = file2.split("_")

    assert len(file1_args) == len(file2_args)

    merged_args = []
    for a1, a2 in zip(file1_args, file2_args):
        if a1.startswith("dv") and a2.startswith("dv"):
            assert a1[2:].split("-")[1] == a2[2:].split("-")[0]
            new_a = a1.split("-")[0] + "-" + a2.split("-")[1]
            merged_args.append(new_a)
        else:
            assert a1 == a2
            merged_args.append(a1)    
    merged_filename = "_".join(merged_args)
    print(merged_filename)
    
    p1 = read_json(file1)
    p2 = read_json(file2)
    dump_json(p1 + p2, merged_filename)    

if __name__ == '__main__':
    merge_predication_chunks(sys.argv[1], sys.argv[2])