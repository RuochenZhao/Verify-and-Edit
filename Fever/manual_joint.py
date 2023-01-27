import os
import argparse

from tqdm import tqdm

from utils import *
from dataset_utils import read_fever_data
from comp_utils import safe_completion, length_of_prompt, conditional_strip_prompt_prefix
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
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=3)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--use_sampled', default=False, action='store_true')

    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/manual{}_sim_{}_tr{}-{}_dv{}-{}_{}_predictions.json".format(args.annotation, args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.style)

def in_context_manual_prediction(ex, training_data, engine, prompt_helper, style="p-e", length_test_only=False):
    prompt, stop_signal = prompt_helper.prompt_for_joint_prediction(ex, training_data)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, temp=0.0, logprobs=5)        

    pred["id"] = ex["id"]
    pred["prompt"] = prompt
    if len(pred["text"]) > len(prompt):
        pred["text"] = pred["text"][len(prompt):]
    else:
        pred["text"] = "null"
    pred["completion_offset"] = len(prompt)
    return pred

def evaluate_manual_predictions(dev_set, predictions, style="p-e", do_print=False):
    acc_records = []
    all_probs = []
    all_texts = []
    for ex, pred in zip(dev_set, predictions):
        gt = ex["label"]
        ex_id = ex['id']
        orig_p = pred["answer"]
        p = normalize_prediction(orig_p)
        all_texts.append(p)
        ex = p == gt        
        acc_records.append(ex)        
        all_probs.append(pred['answer_logprob'])

        if do_print:
            print("--------------EX{}: {}--------------".format(ex_id, ex))
            print(pred["prompt"].split('\n\n')[-1])
            print('P:', p, 'G:', gt)
            print('P RAT:', pred['rationale'])

    print(f'{sum(acc_records)} correct out of {len(acc_records)}')
    print("ACC", sum(acc_records) / len(acc_records))
    
def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_fever_data(f"data/paper_test.jsonl")
    dev_set = dev_set[args.dev_slice:args.num_dev]

    if args.use_sampled:
        sampled_ids = read_json('data/sampled_ids.json')
        dev_set = [[d for d in dev_set if d['id']==id][0] for id in sampled_ids]

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
        if args.use_sampled:
            sampled_ids = read_json('data/sampled_ids.json')
            predictions = [[d for d in predictions if d['id']==id][0] for id in sampled_ids]
    else:
        predictions = []    
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            predictions.append(in_context_manual_prediction(x, train_set, args.engine, args.helper, style=args.style, length_test_only=args.run_length_test))

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'OVER', sum([x > 2048 for x in predictions]))
            return
        # save
        # read un indexed dev
        dump_json(predictions, result_cache_name(args))    
    [args.helper.post_process(p) for p in predictions]
    # acc
    evaluate_manual_predictions(dev_set, predictions, args.style, do_print=True)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)