import sys

import string
import re
from collections import Counter
import numpy as np
import json
from utils import *
import random
import torch

def read_manual_rationale(filename):
    with open(filename) as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        id, rat = l.rstrip().split('\t')
        d[id] = rat
    return d

def read_jsonl(fname):
    with open(fname, 'r') as json_file:
        json_list = list(json_file)
    dev_set = []
    for (i, json_str) in enumerate(json_list):
        try:
            result = json.loads(json_str)
            dev_set.append(result)
        except Exception as e:
            print(i)
            print(json_str)
            print(e)
            raise Exception('end')
    return dev_set

def read_wikiqa_data(fname, manual_annotation_style=None):
    data = read_json(fname)
    if manual_annotation_style is not None:
        man_rat_dict = read_manual_rationale(f'data/manual_rat.txt')
        data = [x for x in data if str(x["_id"]) in man_rat_dict.keys()] # select the data with manual rationales
    else:
        man_rat_dict = dict()
    examples = []
    not_found = 0
    for d in data:
        contexts = d['context']
        supp_pars = []
        for (title, idx) in d['supporting_facts']:
            for c in contexts:
                if c[0]==title:
                    try:
                        supp_pars += [c[1][idx]]
                    except:
                        not_found += 1
        all_pars = [c[1] for c in contexts]
        all_pars = [item for sublist in all_pars for item in sublist]
        ex = {
            "id": d["_id"],
            "question": d["question"].lstrip(),
            "answer": d["answer"],
            'all_pars': all_pars,
            'supp_pars': supp_pars
        }
        if str(ex["id"]) in man_rat_dict.keys():            
            ex["manual_rationale"] = man_rat_dict[str(ex["id"])] #record the manual rationale if exists
        examples.append(ex)   
    print(f'{not_found} not found')
    return examples


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

def wiki_eval(prediction, ground_truth):
    return _exact_match_score(prediction, ground_truth), _f1_score(prediction, ground_truth)

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
    
def wiki_evaluation(prediction, ans):
    ex, fpr = wiki_eval(prediction, ans)
    return  ex, fpr, ans


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


class SimpleRandom():
    instance = None

    def __init__(self,seed):
        self.seed = seed
        self.random = random.Random(seed)

    def next_rand(self,a,b):
        return self.random.randint(a,b)

    @staticmethod
    def get_instance():
        if SimpleRandom.instance is None:
            SimpleRandom.instance = SimpleRandom(SimpleRandom.get_seed())
        return SimpleRandom.instance

    @staticmethod
    def get_seed():
        return int(os.getenv("RANDOM_SEED", 12459))

    @staticmethod
    def set_seeds():

        torch.manual_seed(SimpleRandom.get_seed())
        np.random.seed(SimpleRandom.get_seed())
        random.seed(SimpleRandom.get_seed())

def get_doc_line(lines, line):
    if line > -1:
        return lines.split("\n")[line].split("\t")[1]
    else:
        non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
        return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]




if __name__ == '__main__':
    merge_predication_chunks(sys.argv[1], sys.argv[2])