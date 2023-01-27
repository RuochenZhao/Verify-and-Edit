import os
import argparse

from tqdm import tqdm

from utils import *
from dataset_utils import read_wikiqa_data, f1auc_score, wiki_evaluation
from comp_utils import safe_completion, length_of_prompt, conditional_strip_prompt_prefix
from prompt_helper import get_joint_prompt_helper

# TEST_PART = 250

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
    parser.add_argument('--with_context',  default=False, action='store_true')

    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/manual__tr{}-{}_dv{}-{}_predictions.json".format(
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev)

def in_context_manual_prediction(ex, training_data, engine, prompt_helper, length_test_only=False):
    prompt, stop_signal = prompt_helper.prompt_for_joint_prediction(ex, training_data)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, temp=0.0, logprobs=5)        
        if pred != None:   
            pred["prompt"] = prompt    
            pred["id"] = ex["id"]
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = "null"
            pred["completion_offset"] = len(prompt)
    return pred

def evaluate_manual_predictions(dev_set, predictions, style="p-e", do_print=False):
    acc_records = []
    rat_records = []
    f1_records, pre_records, rec_records = [], [], []
    logprob_records = []
    ansprob_records = []
    
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        p_ans = pred['answer']
        acc, (f1, pre, rec), gt_ans = wiki_evaluation(p_ans, ex["answer"])
        acc_records.append(acc)
        rat_acc = False
        rat_records.append(rat_acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        logprob_records.append(pred['joint_lobprob'])
        ansprob_records.append(pred['answer_logprob'])
        if do_print and not acc:
            print("--------------{} EX {} RAT {} F1 {:.2f}--------------".format(idx, acc, rat_acc, f1))
            print(ex['question'])
            print('\nRAW TEXT', '[' + pred['text'].strip() + ']')
            print('PR ANS:', p_ans)
            print('GT ANS:', gt_ans)
            print(json.dumps({'qas_id': ex['id'], 'answer': p_ans}))

    mean_of_array = lambda x: sum(x) / len(x)
    print("EX", mean_of_array(acc_records), "RAT", mean_of_array(rat_records))
    print("F1: {:.2f}".format(mean_of_array(f1_records)), 
            "PR: {:.2f}".format(mean_of_array(pre_records)),
            "RE: {:.2f}".format(mean_of_array(rec_records)))
    print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
            ansprob_records, acc_records)))
    
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
        for i, x in enumerate(tqdm(dev_set, total=len(dev_set), desc="Predicting")):
            pred = in_context_manual_prediction(x, train_set, engine=args.engine, prompt_helper=args.helper, length_test_only=args.run_length_test)
            if pred != None:
                predictions.append(pred)
            else:
                print('ENDING EARLY')
                args.num_dev = i + args.dev_slice
                dump_json(predictions, result_cache_name(args))
                raise Exception('end')


        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', _MAX_TOKENS)
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