import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_fever_data
from comp_utils import safe_completion, length_of_prompt
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
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=3)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--no_claim',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3.6)
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    parser.add_argument('--no_nei',  default=False, action='store_true')
    
    args = parser.parse_args()
    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    if args.no_claim:
        ending = '_no_claim'
    else:
        ending = ''
    return "misc/verifying_questions_sim_{}_tr{}-{}_dv{}-{}_thres{}_temp_{}{}.json".format( \
        args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.consistency_threshold, args.temperature, ending)

def in_context_manual_prediction(question, sentence, engine, model, helper, no_claim, length_test):
    prompt, stop_signal = helper.prompt_for_question_generation(question, sentence, no_claim = no_claim)
    if length_test:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        return pred
    else:
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
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        if pred['consistency'] < args.consistency_threshold:
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

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_fever_data(f"data/paper_test.jsonl")
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]


    # finished consistency
    predictions = read_full(args, consistency)
    [args.helper.post_process(p) for p in predictions] 
        
    if args.show_prompt:
        prompt, _ = args.helper.prompt_for_question_generation('question', 'sentence', no_claim = False)
        print(prompt)
        raise Exception('prompt shown')

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
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
                    question = in_context_manual_prediction(ex['question'], s, args.engine, args.model, \
                         args.helper, args.no_claim, length_test = args.run_length_test)
                    if args.run_length_test:
                        verifying_questions.append(question)
                    elif question != None:
                        vq.append(question['text'])
                    else:
                        args.num_dev = i + args.dev_slice
                        dump_json(verifying_questions, result_cache_name(args)) 
                        print(result_cache_name(args))
                        raise Exception('end')
                if not args.run_length_test:
                    vq = {'id': ex['id'], 'verifying_questions': vq}
                    verifying_questions.append(vq)
        
        if not args.run_length_test:
            # save
            dump_json(verifying_questions, result_cache_name(args)) 
        else:
            print(result_cache_name(args))
            print('MAX', max(verifying_questions), 'COMP', _MAX_TOKENS)
            print('AVG', sum(verifying_questions)/len(verifying_questions))
            print('TOTAL', sum(verifying_questions))
            print('$', sum(verifying_questions)/1000*0.02)
    evaluate_manual_predictions(dev_set, predictions, verifying_questions, args, do_print=True)
 

if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

