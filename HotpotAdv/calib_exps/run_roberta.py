import sys
from typing import Tuple
sys.path.append('.')

import os
import string
import re
import argparse
import random
from itertools import chain

from tqdm import tqdm

from utils import *
from utils_dataset import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score, train_max_accuracy, f1auc_curve
from utils_comp import safe_completion, length_of_prompt
from ep_baseline import (
    result_cache_name,
    post_process_manual_prediction_and_confidence,
    TEST_PART
)

import spacy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from calib_exps.run_exp import get_evaluation_results, set_seed

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)

    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="e-p")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--sub_calib', type=int, default=64)

    args = parser.parse_args()
    specify_engine(args)
    return args

def load_train_test_split(args):
    examples = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    embeddings = np.load('cls_vecs/hpqa_dev_roberta.npy')    
    for ex, emb in zip(examples, embeddings):
        ex['embedding'] = emb

    examples = examples[args.dev_slice:(args.dev_slice + args.num_dev)]
    predictions = read_json(result_cache_name(args))        
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]

    pairs = list(zip(examples, predictions))

    train_pairs = pairs[:-TEST_PART]
    test_pairs = pairs[-TEST_PART:]
    random.shuffle(train_pairs)

    train_pairs = train_pairs[:(args.sub_calib - args.num_shot)]

    train_pairs = tuple(zip(*train_pairs))
    test_pairs = tuple(zip(*test_pairs))
    return train_pairs, test_pairs

def process_feature_and_label(examples, predictions):

    results_and_indicators = get_evaluation_results(examples, predictions)
    (acc_scores, f1_scores, log_prob_scores,) = [np.array(x) for x in results_and_indicators]        
    embedings = np.asarray([x['embedding'] for x in examples])
    prob_scores = np.exp(log_prob_scores)

    return acc_scores, f1_scores, (prob_scores, embedings)

class RobertaJointCalibrator:
    def __init__(self):
        self.calibrator = LogisticRegression(C=1, max_iter=100)

    def train(self, examples, predictions):
        acc_scores, f1_scores, (prob_scores, embedings) = process_feature_and_label(examples, predictions)
        print(prob_scores.shape, embedings.shape)
        feature = np.concatenate((prob_scores[:,np.newaxis],embedings),1)
        self.calibrator.fit(feature, acc_scores)

    def test(self, examples, predictions):
        acc_scores, f1_scores, (prob_scores, embedings) = process_feature_and_label(examples, predictions)

        print(prob_scores.shape, embedings.shape)
        feature = np.concatenate((prob_scores[:,np.newaxis],embedings),1)
        calib_scores = self.calibrator.predict_proba(feature)[:,1]


        print("By ACC")
        print("P: {:.2f}".format(f1auc_score(
                prob_scores, acc_scores)))
        print("C: {:.2f}".format(f1auc_score(
                calib_scores, acc_scores)))


def calibration_experiment(args):    
    train_pairs, test_pairs = load_train_test_split(args)

    cal = RobertaJointCalibrator()
    cal.train(*train_pairs)
    cal.test(*test_pairs)

if __name__=='__main__':
    set_seed(42)

    args = _parse_args()
    calibration_experiment(args)
