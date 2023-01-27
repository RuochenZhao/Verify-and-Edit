import os
import argparse
from tqdm import tqdm
from utils import *
from comp_utils import safe_completion, length_of_prompt
from dataset_utils import read_fever_data, read_jsonl
import numpy as np
import matplotlib.pyplot as plt
from prompt_helper import get_joint_prompt_helper, normalize_prediction
import pandas as pd
import seaborn as sns
from collections import Counter

_MAX_TOKENS = 144

# PROMPT CONTROL
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)

    parser.add_argument('--style', type=str, default="e-p")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=3)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--plot_consistency',  default=False, action='store_true')
    parser.add_argument('--no_nei',  default=False, action='store_true')
    parser.add_argument('--confidence_threshold', type=float, default=0.98)
    parser.add_argument('--consistency_threshold', type=float, default=3.2)
    parser.add_argument('--use_sampled',  default=False, action='store_true')
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    
    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    ending = ''
    if args.no_nei:
        ending = '_no_nei'
    return "misc/consistency_sim_{}_tr{}-{}_dv{}-{}_predictions_temp_{}{}.json".format(args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.temperature, ending)

def consistency(answers, rationales, predictions, prompt_tokens):
    answer_probs = {}
    answer_prob_lists = {}
    choices = predictions['choices']
    for i, ans in enumerate(answers):
        logprobs = np.array(choices[i]['logprobs']['token_logprobs'][prompt_tokens:])
        prob = np.exp(np.mean(logprobs))
        if ans in answer_probs.keys():
            answer_probs[ans] += prob
            answer_prob_lists[ans] += [(i, prob)]
        else:
            answer_probs[ans] = prob
            answer_prob_lists[ans] = [(i, prob)]
    consistency = max(list(answer_probs.values()))
    final_aggregated_answer = sorted(answer_probs.items(), key=lambda item: item[1], reverse=True)[0][0]
    probs = [a[1] for a in answer_prob_lists[final_aggregated_answer]]
    best_i = np.argmax(probs)
    final_aggregated_rationale = rationales[best_i]
    return consistency, final_aggregated_answer, final_aggregated_rationale, best_i
    
def post_process_consistency(ex, p, args):
    answers, rationales = [], []
    for choice in p['choices']:
        # first split the rationales
        answer, rationale = args.helper.post_process_prediction(choice, no_alter = True)
        answers.append(answer)
        rationales.append(rationale)
    prompt_tokens = p['usage']['prompt_tokens']
    con, best_answer, best_rationale, best_i = consistency(answers, rationales, p, prompt_tokens)
    new_p = p['choices'][best_i]
    new_p['consistency'] = con
    new_p['original_best_rationale'] = best_rationale
    new_p['original_best_answer'] = best_answer
    new_p['original_answers'] = answers
    new_p['original_rationales'] = rationales
    return con, new_p

def in_context_manual_prediction(ex, training_data, engine, prompt_helper, n, model, temp, length_test_only=False):
    prompt, stop_signal = prompt_helper.prompt_for_joint_prediction(ex, training_data)

    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        return pred
    else:
        if model == 'gpt3':
            preds = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = n, temp=temp, logprobs=5)        
    if preds != None:
        for pred in preds['choices']:
            pred["id"] = ex["id"]
            pred["prompt"] = prompt
            if model == 'gpt3':
                if len(pred["text"]) > len(prompt):
                    pred["text"] = pred["text"][len(prompt):]
                else:
                    pred["text"] = "null"
                pred["completion_offset"] = len(prompt)
    return preds

def evaluate_manual_predictions(dev_set, predictions, style="p-e", do_print=False):
    acc_records = []
    all_texts = []

    true_cons = []
    false_cons = []

    true_cert = []
    false_cert = []
    below_conf_thres = []
    below_cons_thres = []
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        gt = ex["label"]
        ex_id = ex['id']

        p_ans = normalize_prediction(pred['answer'])
        all_texts.append(p_ans)

        ex = p_ans == gt        
        acc_records.append(ex)    

        if ex:
            true_cons.append(pred['consistency'])
            true_cert.append(pred['ans_prob_percentage'])
        else:
            false_cons.append(pred['consistency'])
            false_cert.append(pred['ans_prob_percentage'])
        if pred['ans_prob_percentage'] < args.confidence_threshold:
            below_conf_thres.append(ex)
        if pred['consistency'] < args.consistency_threshold:
            below_cons_thres.append(ex)

        if do_print:
            print("--------------{} EX {} CONS {:.2f}--------------".format(ex_id, ex, pred['consistency']))
            print(pred["prompt"].split('\n\n')[-1])
            for (i, answer) in enumerate(pred['original_answers']):
                rat = pred['original_rationales'][i]
                print(f'{i}: {rat} | {answer}')
            print('P RAT:', pred['rationale'])
            print('P:', p_ans, 'G:', gt)

    print(f'{sum(acc_records)} correct out of {len(acc_records)}')
    print("ACC", sum(acc_records) / len(acc_records))

    cons = true_cons + false_cons
    print('consistencies: mean {} and std {}'.format(np.mean(cons), np.std(cons)))
    
    print('consistencies for true predictions: mean {} and std {}'.format(np.mean(true_cons), np.std(true_cons)))
    print('consistencies for false predictions: mean {} and std {}'.format(np.mean(false_cons), np.std(false_cons)))
    print('below confidence threshold: ', Counter(below_conf_thres))
    print('below consistency threshold: ', Counter(below_cons_thres))
    return (true_cons, false_cons), (true_cert, false_cert)

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_jsonl(f"data/paper_test_processed.jsonl")
    dev_set = dev_set[args.dev_slice:args.num_dev]
    if args.no_nei:
        dev_set = [d for d in dev_set if d['label'] != 'NOT ENOUGH INFO']
    if args.use_sampled:
        sampled_ids = read_json('data/sampled_ids.json')
        dev_set = [d for d in dev_set if d['id'] in sampled_ids]
    if args.show_prompt:
        prompt, _ = args.helper.prompt_for_joint_prediction(dev_set[0], train_set)
        print(prompt)
        raise Exception('prompt shown')

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
        if args.use_sampled:
            predictions = [d for d in predictions if d['id'] in sampled_ids]
    else:
        predictions = []    
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            pred = in_context_manual_prediction(x, train_set, args.engine, args.helper, 5, args.model, \
                args.temperature, length_test_only=args.run_length_test)
            if pred != None:
                predictions.append(pred)
            else: #error, ending early
                args.num_dev = len(predictions) + args.dev_slice
                break
                
        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', _MAX_TOKENS)
            print('TOTAL', sum(predictions))
            return
            
        # save
        # read un indexed dev
        dump_json(predictions, result_cache_name(args))   
    
        new_predictions, cons = [], []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            con, new_p = post_process_consistency(ex, p, args)
            cons.append(con)
            new_predictions.append(new_p)
        dump_json(new_predictions, result_cache_name(args)) 
        predictions = new_predictions 

    
    [args.helper.post_process(p) for p in predictions]
    # acc
    (true_cons, false_cons), (true_cert, false_cert) = evaluate_manual_predictions(dev_set, predictions, args.style, do_print=True)
    print(result_cache_name(args))

    cons = [p['consistency'] for p in predictions]
    plt.figure(figsize=(10,5))
    df = pd.DataFrame.from_dict({'label': ['correct']*len(true_cons) + ['incorrect']*len(false_cons) + ['overall']*(len(true_cons)+len(false_cons))\
        , 'consistency': true_cons + false_cons + cons})
    sns.histplot(data=df, x="consistency", hue="label")
    plt.savefig(f"log/consistency_{args.engine_name}.png")

    cert = true_cert + false_cert
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    df = pd.DataFrame.from_dict({'label': ['correct']*len(true_cert) + ['incorrect']*len(false_cert) + ['overall']*(len(true_cert)+len(false_cert))\
        , 'confidence': true_cert + false_cert + cert})
    sns.histplot(data=df, x="confidence", hue="label", bins=10)
    plt.savefig(f"log/confidence_{args.engine_name}.png")

if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')