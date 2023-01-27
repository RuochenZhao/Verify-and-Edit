import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_wikiqa_data
from comp_utils import safe_completion
from prompt_helper import get_joint_prompt_helper
import consistency

_MAX_TOKENS = 70

# PROMOT CONTROL
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]

def _parse_args():
    '''
    Function that parses arguments passed to the script
    '''
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
    parser.add_argument('--consistency_threshold', type=float, default=3)
    
    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/verifying_questions_tr{}-{}_dv{}-{}_thres{}_temp_{}.json".format( \
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.consistency_threshold, args.temperature)

def in_context_manual_prediction(question, sentence, engine, model, helper):
    prompt, stop_signal = helper.prompt_for_question_generation(question, sentence)
    if model == 'gpt3':
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = 1, temp=0.0, logprobs=5)        
        if pred != None:
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = "null"
            pred["completion_offset"] = len(prompt)
    return pred

def evaluate_manual_predictions(dev_set, predictions, verifying_questions, args, do_print=False):
    num = 0
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        if pred['consistency'] < args.consistency_threshold:
            num += 1
            id = ex['id']
            for c in verifying_questions:
                if c['id']==id:
                    cont = c['verifying_questions']
                    break
            if do_print:
                print("--------------{} EX {} CONS--------------".format(idx, pred['consistency']))
                print('question: ', ex['question'])
                sentences = rationale_tokenize(pred['rationale'])

                for j, s in enumerate(sentences):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, cont[j]))
    print(f'{num} instances below consistency threshold')

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_wikiqa_data(f"data/train_subset.json", manual_annotation_style=args.style)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    print('len(train_set): ', len(train_set))
    dev_set = read_wikiqa_data(f"data/dev_sampled.json")
    dev_set = dev_set[args.dev_slice:(args.num_dev)]

    prompt, _ = args.helper.prompt_for_question_generation('question', 'sentence')
    print('prompt: ')
    print(prompt)

    # finished consistency and processs
    print('args.num_dev: ', args.num_dev)
    predictions = read_full(args, consistency)
    new_predictions, cons = [], []
    for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
        ex = dev_set[i]
        con, new_p = consistency.post_process_consistency(ex, p, args)
        cons.append(con)
        new_predictions.append(new_p)
    predictions = new_predictions 
    [args.helper.post_process(p) for p in predictions] 

    if os.path.exists(result_cache_name(args)):
        # finished all steps, evaluating
        verifying_questions = read_json(result_cache_name(args))
        print(result_cache_name(args))
    else:
        print('running verifying question generation')
        verifying_questions = []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            con = p['consistency']
            if con < args.consistency_threshold:
                vq = []
                sentences = rationale_tokenize(p['rationale'])
                for q, s in enumerate(sentences):
                    question = in_context_manual_prediction(ex['question'], s, args.engine, args.model, args.helper)
                    if question != None:
                        vq.append(question['text'])
                    else:
                        args.num_dev = i + args.dev_slice
                        dump_json(verifying_questions, result_cache_name(args)) 
                        print(result_cache_name(args))
                        raise Exception('end')
                vq = {'id': ex['id'], 'verifying_questions': vq}
                verifying_questions.append(vq)
        # save
        dump_json(verifying_questions, result_cache_name(args)) 
        
    evaluate_manual_predictions(dev_set, predictions, verifying_questions, args, do_print=True)
 

if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

