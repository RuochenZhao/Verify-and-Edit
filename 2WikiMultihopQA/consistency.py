import os
import argparse
from tqdm import tqdm
from utils import *
from comp_utils import safe_completion, length_of_prompt
from dataset_utils import read_wikiqa_data, f1auc_score, wiki_evaluation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from prompt_helper import get_joint_prompt_helper, normalize_prediction

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
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=1000)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--plot_consistency',  default=False, action='store_true')
    
    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/consistency_tr{}-{}_dv{}-{}_predictions_temp_{}.json".format(
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.temperature)

def consistency(answers, rationales, predictions, prompt_tokens):
    answer_probs = {}
    answer_prob_lists = {}
    choices = predictions['choices']
    for i, ans in enumerate(answers):
        logprobs = np.array(choices[i]['logprobs']['token_logprobs'][prompt_tokens:])
        prob = np.exp(np.mean(logprobs))
        if ans in answer_probs.keys():
            answer_prob_lists[ans].append((i, prob))
            answer_probs[ans] += 1
        else:
            answer_prob_lists[ans] = [(i, prob)]
            answer_probs[ans] = 1
    consistency = max(list(answer_probs.values()))/5
    final_aggregated_answer = sorted(answer_probs.items(), key=lambda item: item[1], reverse=True)[0][0]
    prob_list = answer_prob_lists[final_aggregated_answer]
    best_i = prob_list[np.argmax([a[1] for a in prob_list])][0]
    final_aggregated_rationale = rationales[best_i]
    return consistency, final_aggregated_answer, final_aggregated_rationale, best_i

def post_process_consistency(ex, p, args):
    answers, rationales = [], []
    for choice in p['choices']:
        # first split the rationales
        answer, rationale = args.helper.post_process_prediction(choice, change_rationale = False)
        answers.append(answer)
        rationales.append(rationale)
    prompt_tokens = p['usage']['prompt_tokens']
    con, best_answer, best_rationale, best_i = consistency(answers, rationales, p, prompt_tokens)
    new_p = p['choices'][best_i]
    new_p['id'] = ex['id']
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
        print(prompt)
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
    rat_records = []
    f1_records, pre_records, rec_records = [], [], []
    logprob_records = []
    ansprob_records = []

    true_cons = []
    false_cons = []
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        p_ans = pred['answer']
        p_rat = pred['rationale']
        acc, (f1, pre, rec), gt_ans = wiki_evaluation(p_ans, ex["answer"])
        acc_records.append(acc)
        rat_acc = False
        rat_records.append(rat_acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        logprob_records.append(pred['joint_lobprob'])
        ansprob_records.append(pred['answer_logprob'])
        if acc:
            true_cons.append(pred['consistency'])
        else:
            false_cons.append(pred['consistency'])

        if do_print:
            print("--------------{} EX {} RAT {} F1 {:.2f} CONS {:.2f}--------------".format(idx, acc, rat_acc, f1, pred['consistency']))
            print('question: ', ex['question'])
            for (i, answer) in enumerate(pred['original_answers']):
                rat = pred['original_rationales'][i]
                print(f'{i}: {rat} | {answer}')
            print('PR ANS:', p_ans)
            print('PR RAT:', p_rat)
            print('GT ANS:', gt_ans)
            print(json.dumps({'qas_id': ex['id'], 'answer': p_ans}))

    mean_of_array = lambda x: sum(x) / len(x)
    print("EX", mean_of_array(acc_records), "RAT", mean_of_array(rat_records))
    print("F1: {:.2f}".format(mean_of_array(f1_records)), 
            "PR: {:.2f}".format(mean_of_array(pre_records)),
            "RE: {:.2f}".format(mean_of_array(rec_records)))
    print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
            ansprob_records, acc_records)))
    
    cons = true_cons + false_cons
    print('consistencies: mean {} and std {}'.format(np.mean(cons), np.std(cons)))
    
    print('consistencies for true predictions: mean {} and std {}'.format(np.mean(true_cons), np.std(true_cons)))
    print('consistencies for false predictions: mean {} and std {}'.format(np.mean(false_cons), np.std(false_cons)))
    return true_cons, false_cons

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_wikiqa_data(f"data/train_subset.json", manual_annotation_style=args.style)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    print('len(train_set): ', len(train_set))
    dev_set = read_wikiqa_data(f"data/dev_sampled.json")
    dev_set = dev_set[args.dev_slice:(args.num_dev)]

    prompt, _ = args.helper.prompt_for_joint_prediction(dev_set[0], train_set)
    print('prompt: ')
    print(prompt)

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
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
            return
            
        # save
        dump_json(predictions, result_cache_name(args))   
    
    new_predictions, cons = [], []
    for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
        ex = dev_set[i]
        con, new_p = post_process_consistency(ex, p, args)
        cons.append(con)
        new_predictions.append(new_p)
    predictions = new_predictions 

    
    [args.helper.post_process(p) for p in predictions]
    # acc
    true_cons, false_cons = evaluate_manual_predictions(dev_set, predictions, args.style, do_print=True)
    print(result_cache_name(args))

    cons = [p['consistency'] for p in predictions]
    plt.figure(figsize=(10,5))
    df = pd.DataFrame.from_dict({'label': ['correct']*len(true_cons) + ['incorrect']*len(false_cons) + ['overall']*(len(true_cons)+len(false_cons))\
        , 'consistency': true_cons + false_cons + cons})
    sns.displot(df, x="consistency", hue="label", kind="kde", fill=True)
    plt.savefig(f"log/consistency_{args.engine_name}.png")
    

if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')