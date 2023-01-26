import argparse
import os
from tqdm import tqdm

from utils import *
from dataset_utils import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score, read_incorrect_answers
from comp_utils import safe_completion, length_of_prompt
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308) # firs 58 for calibrating, last 250 for testing
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    args = parser.parse_args()    
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/few_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.json".format(args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.dev_slice + args.num_dev,
        args.num_distractor, args.style)

def convert_paragraphs_to_context(s, connction='\n'):
    return connction.join(['{}'.format(p) for i, p in enumerate(s['pars'])])

def in_context_prediction(context, ex, shots, engine, style="standard", length_test_only=False):
    if style == "standard":
        if context:
            showcase_examples = [
                "{}\nQ: {}\nA: {}\n".format(convert_paragraphs_to_context(s), s["question"], s["answer"]) for s in shots
            ]
            input_example = "{}\nQ: {}\nA:".format(convert_paragraphs_to_context(ex), ex["question"])
        else:
            showcase_examples = [
                "Q: {}\nA: {}\n".format(s["question"], s["answer"]) for s in shots
            ]
            input_example = "Q: {}\nA:".format(ex["question"])
        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")
    if length_test_only:
        pred = length_of_prompt(prompt, 32)
        print("-----------------------------------------")
        print(pred)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, 32, '\n', temp=0.0, logprobs=5)    

    pred["id"] = ex["id"]
    pred["prompt"] = prompt
    try:
        if len(pred["text"]) > len(prompt):
            pred["text"] = pred["text"][len(prompt):]
        else:
            pred["text"] = "null"
        return pred
    except:
        return None

def evaluate_few_shot_predictions(dev_set, predictions, do_print=False):
    acc_records = []    
    f1_records, pre_records, rec_records = [], [], []
    logprob_records = []
    
    certified_incorrect_answers = read_incorrect_answers()
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        gt_rat = ' '.join(ex['rationale'])
        p_ans = pred['text'].lstrip()
        acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
        acc_records.append(acc)                
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        if 'answer_prob' in pred:
            logprob_records.append(pred['answer_prob'])
        if do_print and not acc:
            if ex['id'] in certified_incorrect_answers and p_ans in certified_incorrect_answers[ex['id']]:
                continue
            print("--------------{} EX {} F1 {:.2f}--------------".format(idx, acc, f1))
            print(convert_paragraphs_to_context(ex))
            print(ex['question'])

            print('PR ANS:', p_ans)            
            print('GT ANS:', gt_ans)            
            print(json.dumps({'qas_id': ex['id'], 'answer': p_ans}))

    mean_of_array = lambda x: sum(x) / len(x)
    print("EX", mean_of_array(acc_records))
    print("F1: {:.2f}".format(mean_of_array(f1_records)), 
            "PR: {:.2f}".format(mean_of_array(pre_records)),
            "RE: {:.2f}".format(mean_of_array(rec_records)))
    print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
            logprob_records, acc_records)))

def test_few_shot_performance(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", n_dist=args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    if args.show_prompt:
        showcase_examples = [
            "Q: {}\nA: {}\n".format(s["question"], s["answer"]) for s in train_set
        ]
        prompt = "\n".join(showcase_examples)
        print(prompt)
        raise Exception('prompt shown')

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        predictions = []
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            pred = in_context_prediction(args.with_context, x, train_set, engine=args.engine, \
                style=args.style, length_test_only=args.run_length_test)
            if pred == None:
                args.num_dev = len(predictions)
                break
            else:
                predictions.append(pred)

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', 32)
            return
        # save
        dump_json(predictions, result_cache_name(args))
    # acc
    for p in predictions:
        p['answer_prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    analyze_few_shot_performance(args)


def calc_fewshot_pred_with_prob(pred, style):
    if pred['text'] == "null" or pred['text'] == "overflow":
        print("find invalid", pred["text"])
        return .0
    completion_offset = len(pred["prompt"])
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    completion_end_tok_idx = tokens.index("<|endoftext|>") + 1 if '<|endoftext|>' in tokens else len(tokens)
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]
    ans_logprob = sum(completion_probs)

    return np.exp(ans_logprob)

def analyze_few_shot_performance(args):
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]        
    predictions = read_json(result_cache_name(args))

    if args.show_result:
        dev_set = dev_set
        predictions = predictions

    for p in predictions:
        p['answer_prob'] = calc_fewshot_pred_with_prob(p, args.style) 
    print(len(predictions))
    print(result_cache_name(args))
    evaluate_few_shot_predictions(dev_set, predictions, do_print=True)

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        analyze_few_shot_performance(args)