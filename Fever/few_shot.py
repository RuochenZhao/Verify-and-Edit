import argparse
import os
from tqdm import tqdm
from prompt_helper import normalize_prediction
from utils import *
from dataset_utils import read_fever_data
from comp_utils import safe_completion, length_of_prompt
import numpy as np

TEST_PART = 250
_PROMPT_HEADER = 'Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFO.'

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=3)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300) # firs 58 for calibrating, last 250 for testing
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--use_sampled',  default=False, action='store_true')
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    args = parser.parse_args()    
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/few_sim_{}_tr{}-{}_dv{}-{}_{}_predictions.json".format(args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.style)

def convert_paragraphs_to_context(s, connction='\n'):
    return connction.join(['{}'.format(p) for i, p in enumerate(s['pars'])])

def calc_fewshot_pred_with_prob(pred, style):
    if pred['text'] == "null" or pred['text'] == "overflow":
        return .0
    completion_offset = len(pred["prompt"])
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    completion_end_tok_idx = tokens.index("<|endoftext|>") + 1 if '<|endoftext|>' in tokens else len(tokens)
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]    
    ans_logprob = sum(completion_probs)
    
    return np.exp(ans_logprob)

def calc_fewshot_cls_prob(pred, style):
    if pred['text'] == "null" or pred['text'] == "overflow":
        pred['class_probs'] = [.0, .0, 1.0]
    completion_offset = len(pred["prompt"])
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    completion_end_tok_idx = tokens.index("<|endoftext|>") + 1 if '<|endoftext|>' in tokens else len(tokens)

    top_choices = pred["logprobs"]["top_logprobs"][completion_start_tok_idx]    
    if style == 'standard':
        mappings = [' SUPPORTS', ' REFUTES', ' NOT ENOUGH INFO']
    else:
        raise RuntimeError("Unsupported Style")
    cls_probs = []
    for t in mappings:
        if t in top_choices:
            cls_probs.append(np.exp(top_choices[t]))
        else:
            cls_probs.append(.0)    
    pred['class_probs'] = cls_probs

def post_process_fewshot_prediction(p, style):
    p['prob'] = calc_fewshot_pred_with_prob(p, style)
    calc_fewshot_cls_prob(p, style)
    p['label'] = normalize_prediction(p['text'])

def in_context_prediction(ex, shots, engine, style="standard", length_test_only=False):
    if style == "standard":
        showcase_examples = [
            "Claim: {}\nA: {}\n".format(s["question"], s["label"]) for s in shots
        ]
        input_example = "Claim: {}\nA:".format(ex["question"])
        
        prompt = "\n".join(showcase_examples + [input_example])
        prompt = _PROMPT_HEADER + '\n\n' + prompt
    else:
        raise RuntimeError("Unsupported prompt style")
    if length_test_only:
        pred = length_of_prompt(prompt, 32)
        print("-----------------------------------------")
        print(pred)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, 20, '\n', temp=0.0, logprobs=5)    

    pred["id"] = ex["id"]
    pred["prompt"] = prompt
    if len(pred["text"]) > len(prompt):
        pred["text"] = pred["text"][len(prompt):]
    else:
        pred["text"] = "null"
    pred["completion_offset"] = len(prompt)
    return pred

def evaluate_few_shot_predictions(dev_set, predictions, do_print=False):
    acc_records = []
    all_texts = []
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        gt = ex["label"]
        orig_p = pred["text"]
        p = normalize_prediction(orig_p)
        all_texts.append(p)
        ex = p == gt        
        acc_records.append(ex)

        if do_print:
            print("--------------{} EX {}--------------".format(pred['id'], ex))
            print(pred["prompt"].split('\n\n')[-1])
            print('RAW:', orig_p)
            print('P:', p, 'G:', gt)
            
    print(f'{sum(acc_records)} correct out of {len(acc_records)}')
    print("ACC", sum(acc_records) / len(acc_records)) 

def test_few_shot_performance(args):
    print("Running prediction")
    train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_fever_data(f"data/paper_test.jsonl")
    dev_set = dev_set[args.dev_slice:args.num_dev]
    if args.use_sampled:
        sampled_ids = read_json('data/sampled_ids.json')
        dev_set = [[d for d in dev_set if d['id']==id][0] for id in sampled_ids]
    if args.show_prompt:
        showcase_examples = [
            "Claim: {}\nA: {}\n".format(s["question"], s["label"]) for s in train_set
        ]
        prompt = "\n".join(showcase_examples)
        prompt = _PROMPT_HEADER + '\n\n' + prompt
        print(prompt)
        raise Exception('prompt shown')

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
        if args.use_sampled:
            predictions = [[d for d in predictions if d['id']==id][0] for id in sampled_ids]
    else:
        predictions = []
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            predictions.append(in_context_prediction(x, train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test))

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', 12)
            return
        # save
        dump_json(predictions, result_cache_name(args))
    # acc
    for p in predictions:
        post_process_fewshot_prediction(p, args.style)
    evaluate_few_shot_predictions(dev_set, predictions, do_print=True)


def calc_fewshot_pred_with_prob(pred, style):
    if pred['text'] == "null" or pred['text'] == "overflow":
        print("find invalid", pred["text"])
        return .0
    completion_offset = len(pred["prompt"])
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    completion_end_tok_idx = tokens.index("<|endoftext|>") + 1 if '<|endoftext|>' in tokens else len(tokens)
    completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]
    ans_logprob = sum(completion_probs)

    return np.exp(ans_logprob)

def analyze_few_shot_performance(args):
    dev_set = read_fever_data(f"data/paper_test.jsonl")
    dev_set = dev_set[args.dev_slice:args.num_dev]        
    predictions = read_json(result_cache_name(args))

    if args.show_result:
        dev_set = dev_set[-TEST_PART:]
        predictions = predictions[-TEST_PART:]

    for p in predictions:
        post_process_fewshot_prediction(p, args.style)
    print(len(predictions))
    print(result_cache_name(args))
    evaluate_few_shot_predictions(dev_set, predictions, do_print=True)

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        analyze_few_shot_performance(args)