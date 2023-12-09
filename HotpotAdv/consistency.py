import os
import argparse
from tqdm import tqdm
from utils import *
from comp_utils import safe_completion, length_of_prompt
from few_shot import convert_paragraphs_to_context
from dataset_utils import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score, read_incorrect_answers, normalize_answer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


_MAX_TOKENS = 144

# PROMOT CONTROL
PE_STYLE_SEP = " The reason is as follows."
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
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--plot_consistency',  default=False, action='store_true')
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    
    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/consistency_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.json".format(args.annotation, args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.num_distractor, args.style)

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
        answer, rationale = post_process_manual_prediction(choice, args.style, no_alter = True)
        answers.append(answer)
        rationales.append(rationale)
    prompt_tokens = p['usage']['prompt_tokens']
    con, best_answer, best_rationale, best_i = consistency(answers, rationales, p, prompt_tokens, args.model)
    new_p = p['choices'][best_i]
    new_p['consistency'] = con
    new_p['original_best_rationale'] = best_rationale
    new_p['original_best_answer'] = best_answer
    new_p['original_answers'] = answers
    new_p['original_rationales'] = rationales
    return con, new_p

def in_context_manual_prediction(ex, training_data, engine, n, model, temp, with_context, style="p-e", length_test_only=False, \
                                 gptj_model = None, device = None):
    prompt, stop_signal = prompt_for_manual_prediction(ex, training_data, style, with_context)

    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        # print(prompt)
        return pred
    else:
        if model == 'gpt3':
            preds = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = n, temp=0.0, logprobs=5)        
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

def post_process_manual_prediction(p, style, no_alter = False):
    text = p["text"]
    text = text.strip()

    # place holder
    answer = "null"
    rationale = "null"
    rationale_indices = []
    if style == "p-e":         
        sep = PE_STYLE_SEP
        if sep in text:
            segments = text.split(sep)   
            answer = segments[0].strip().strip('.')
            rationale = segments[1].strip()
    elif style == "e-p":
        sep = get_sep_text(p, style)
        if sep is not None:
            segments = text.split(sep)
            try:
                answer = segments[1].strip().strip('.')
                rationale = segments[0].strip()
            except:
                print('sep: ', sep)
                print('text: ', text)
                print('invalid segment: ', segments)
        else:
            answer = text
    else:
        raise RuntimeError("Unsupported decoding style")
    if no_alter == False: #alter it
        p["answer"] = answer
        p["rationale"] = rationale
        p["rationale_indices"] = rationale_indices
    return answer, rationale


def post_process_manual_confidance(pred, style):
    completion_offset = pred["completion_offset"]
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    # exclusive idxs
    if "<|endoftext|>" in tokens:
        completion_end_tok_idx = tokens.index("<|endoftext|>") + 1
    else:
        completion_end_tok_idx = len(tokens)
    completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]

    if style == "p-e":            
        if PE_STYLE_SEP in pred["text"]:
            sep_token_offset = completion_offset + pred["text"].index(PE_STYLE_SEP)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            ans_logprob = sum(completion_probs[:sep_start_idx - 1])
            rat_logprob = sum(completion_probs[(sep_start_idx + 6):])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0
    elif style == "e-p":
        sep_text = get_sep_text(pred, style)
        if sep_text is not None:
            sep_token_offset = completion_offset + pred["text"].index(sep_text)
            if sep_token_offset not in token_offset:
                # sometimes fail to parse special characters
                sep_token_offset = min([to for to in token_offset if to>sep_token_offset])
                print(f'SPECIAL CASE -- TOKEN {pred["text"][sep_token_offset-1:]} SEP_TEXT {sep_text}')
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            rat_logprob = sum(completion_probs[:sep_start_idx + 3])
            ans_logprob = sum(completion_probs[(sep_start_idx + 3):-1])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0
    else:
        raise RuntimeError("Unsupported decoding style")

    pred["answer_logprob"] = ans_logprob
    pred["rationale_logprob"] = rat_logprob
    pred["joint_lobprob"] = ans_logprob + rat_logprob
    return ans_logprob, rat_logprob

def post_process_manual_prediction_and_confidence(pred, style):
    # process answer and rationale
    post_process_manual_prediction(pred, style)
    post_process_manual_confidance(pred, style)

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
        acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
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
            print(convert_paragraphs_to_context(ex))
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
    print('consistencies for true predictions: ', true_cons)
    print('consistencies for false predictions: ', false_cons)
    return true_cons, false_cons

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:args.num_dev]
    if args.show_prompt:
        prompt, stop_signal = prompt_for_manual_prediction(dev_set[0], train_set, args.style, args.with_context)
        print(prompt)
        raise Exception('prompt shown')
    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        # raise Exception(f'{result_cache_name(args)} not found')
        predictions = []    
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            pred = in_context_manual_prediction(x, train_set, args.engine, 5, args.model, \
                args.temperature, args.with_context, style="e-p", \
                length_test_only=args.run_length_test)
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

    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]
    # acc
    true_cons, false_cons = evaluate_manual_predictions(dev_set, predictions, args.style, do_print=True)
    
    # if args.plot_consistency:
    cons = [p['consistency'] for p in predictions]
    plt.figure(figsize=(10,5))
    df = pd.DataFrame.from_dict({'label': ['correct']*len(true_cons) + ['incorrect']*len(false_cons) + ['overall']*(len(true_cons)+len(false_cons))\
        , 'consistency': true_cons + false_cons + cons})
    sns.displot(df, x="consistency", hue="label", kind="kde", fill=True)
    plt.savefig(f"log/consistency_{args.engine_name}.png")
    
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')