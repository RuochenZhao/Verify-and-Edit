import sys
sys.path.append('.')
import string
import re
import argparse
import random

from utils import *
from dataset_utils import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score
from consistency import (
    result_cache_name,
    post_process_manual_prediction_and_confidence,
)
from manual_joint import TEST_PART
import spacy
import numpy as np

from sklearn.linear_model import LogisticRegression

nlp = spacy.load('en_core_web_sm')

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
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)

    args = parser.parse_args()
    specify_engine(args)
    return args

def extract_stem_tokens(text):
    doc = nlp(text)
    stem_tokens = []
    for i, t in enumerate(doc):
        pos, tag = t.pos_, t.tag_
        if pos == 'AUX':
            continue
        is_stem = False
        if tag.startswith('NN'):
            is_stem = True
        if tag.startswith('VB'):
            is_stem = True
        if tag == 'CD':
            is_stem = True
        if is_stem:
            stem_tokens.append({
                'index': i,
                'text': t.text,
                'lemma': t.lemma_,
                'pos': t.pos_,
                'tag': t.tag_
            })        
    return stem_tokens


def get_evaluation_results(examples, predictions):
    acc_records = []
    f1_records, pre_records, rec_records = [], [], []
    logprob_records = []
    for ex, pred in zip(examples, predictions):
        gt_ans = ex['answer']
        gt_rat = ' '.join(ex['rationale'])
        p_ans = pred['answer']
        p_rat = pred['rationale']
        acc, (f1, pre, rec), _ = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
        acc_records.append(acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        logprob_records.append(pred['answer_logprob'])        

    mean_of_array = lambda x: sum(x) / len(x)
    # print("EX", mean_of_array(acc_records))
    # print("F1: {:.2f}".format(mean_of_array(f1_records)), 
    #         "PR: {:.2f}".format(mean_of_array(pre_records)),
    #         "RE: {:.2f}".format(mean_of_array(rec_records)))
    return acc_records, f1_records, logprob_records

def maximum_common_substrings(text1, text2):
    len1 = len(text1)
    len2 = len(text2)
    table = [[  tuple() for i in range(len2 + 1)] for j in range(len1 + 1)]

    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0 or j == 0:
                table[i][j] = (0,[])
                continue
            if text1[i - 1] == text2[j - 1]:
                max_len, cur_seq = table[i - 1][j - 1]
                table[i][j] = (max_len + 1, cur_seq + [text1[i-1]])
            else:
                if table[i - 1][j][0] >= table[i][j - 1][0]:                    
                    table[i][j] = table[i - 1][j ]
                else:                    
                    table[i][j] = table[i][j - 1]
    return table[-1][-1]

def product_of_list(l):
    v = 1
    for x in l:
        v = v * x
    return v

def rationale_validness_quality(rationale, pars, question, answer, cutoff_threshold=0.9):
    rationale = rationale.replace("First,", "<S>").replace("Second,", "<S>").replace("Third,", "<S>").replace("Fourth,", "<S>")
    claims = rationale.split("<S>")
    claims = [x.strip() for x in claims if x.strip()]

    stem_toks_by_claims = [ [t['lemma'] for t in extract_stem_tokens(c)] for c in claims]
    stem_toks_by_paragraphs = [ [t['lemma'] for t in extract_stem_tokens(c)] for c in pars]
    stem_toks_of_question = [t['lemma'] for t in extract_stem_tokens(question)]
    stem_toks_of_answers = [t['lemma'] for t in extract_stem_tokens(answer)]

    valid_stem_toks_of_question = set(stem_toks_of_question)


    val_of_claims = []

    for s_c in stem_toks_by_claims:        
        if not s_c:
            val_of_claims.append(1.0)
            continue
        max_sup = 0
        max_over = ([],[])
        for s_p in stem_toks_by_paragraphs:
            # sup = len([x for x in s_c if x in s_p])
            sup, max_com = maximum_common_substrings(s_c, s_p)
            if sup > max_sup:
                max_sup = sup
                max_over = s_p, max_com

        sup, _ = maximum_common_substrings(s_c, stem_toks_of_question + stem_toks_of_answers)
        if sup > max_sup:
            max_sup = sup
        val_of_claims.append(max_sup / len(s_c))

    return min(val_of_claims) if val_of_claims else 0
    

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


def load_train_test_split(args):
    examples = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    examples = examples[args.dev_slice:(args.dev_slice + args.num_dev)]
    predictions = read_json(result_cache_name(args))
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]

    pairs = list(zip(examples, predictions))

    train_pairs = pairs[:-TEST_PART]
    test_pairs = pairs[-TEST_PART:]

    random.shuffle(train_pairs)
    train_pairs = train_pairs[:(args.sub_calib - 6)]

    train_pairs = tuple(zip(*train_pairs))
    test_pairs = tuple(zip(*test_pairs))
    return train_pairs, test_pairs

def process_feature_and_label(examples, predictions):

    for ex, pred in zip(examples, predictions):        
        rat_val = rationale_validness_quality(pred['rationale'], ex['pars'], ex['question'], ex['answer'])
        pred['rationale_validness'] = rat_val            

    results_and_indicators = get_evaluation_results(examples, predictions)
    (acc_scores, f1_scores, log_prob_scores,) = [np.array(x) for x in results_and_indicators]        
    validness_scores = np.array([x['rationale_validness'] for x in predictions])
    validness_scores = np.power(validness_scores, 3)
    prob_scores = np.exp(log_prob_scores)

    return acc_scores, f1_scores, (prob_scores, validness_scores)

class Calibrator:
    def __init__(self):
        self.calibrator = LogisticRegression(C=10, max_iter=100)

    def train(self, examples, predictions):
        acc_scores, f1_scores, feat_groups = process_feature_and_label(examples, predictions)
        feature = np.stack(feat_groups).transpose()
        self.calibrator.fit(feature, acc_scores)

    def test(self, examples, predictions):
        acc_scores, f1_scores, feat_groups = process_feature_and_label(examples, predictions)
        prob_scores = feat_groups[0]
        feature = np.stack(feat_groups).transpose()
        calib_scores = self.calibrator.predict_proba(feature)[:,1]

        print("AUC")
        print("P: {:.2f}".format(f1auc_score(
                prob_scores, acc_scores)))
        print("C: {:.2f}".format(f1auc_score(
                calib_scores, acc_scores)))

def calibration_experiment(args):    
    train_pairs, test_pairs = load_train_test_split(args)
    cal = Calibrator()
    cal.train(*train_pairs)
    cal.test(*test_pairs)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
        
if __name__=='__main__':
    set_seed(42)

    args = _parse_args()
    calibration_experiment(args)
