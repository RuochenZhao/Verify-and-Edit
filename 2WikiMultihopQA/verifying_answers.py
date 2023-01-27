import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_wikiqa_data
from comp_utils import safe_completion, length_of_prompt
from prompt_helper import get_joint_prompt_helper
import consistency
import verifying_questions
import relevant_context

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
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3)
    # for relevant_context retrieval
    parser.add_argument('--retrieval', choices=['wikipedia', 'drqa', 'google', 'dataset'])
    
    args = parser.parse_args()

    args.helper = get_joint_prompt_helper(args.style)
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/verifying_answers_tr{}-{}_dv{}-{}_{}_thres{}.json".format( \
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.retrieval, args.consistency_threshold)

def in_context_manual_prediction(shots, engine, prompt_helper, model, pars, verifying_question):
    prompt, stop_signal = prompt_helper.prompt_for_verifying_answers(shots, pars, verifying_question)
    if model == 'gpt3':
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = 1, temp=0.0, logprobs=5)        
        if pred != None:
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = "null"
            pred["completion_offset"] = len(prompt)
    return pred

def evaluate_manual_predictions(dev_set, predictions, verifying_qs, contexts, verifying_answers, args, do_print=False):
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        if pred['consistency'] < args.consistency_threshold:
            id = ex['id']
            vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
            cont = [c for c in contexts if c['id']==id][0]['context']
            va = [c for c in verifying_answers if c['id']==id][0]['verifying_answers']
            if do_print:
                print("--------------{} EX {} CONS--------------".format(id, pred['consistency']))
                print('question: ', ex['question'])
                sentences = rationale_tokenize(pred['rationale'])
                for j, (s, q, c, a) in enumerate(zip(sentences, vq, cont, va)):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, q))
                    print('contexts {}: {}'.format(j, c))
                    print('verifying_answers {}: {}'.format(j, a))

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_wikiqa_data(f"data/train_subset.json", manual_annotation_style=args.style)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    print('len(train_set): ', len(train_set))
    dev_set = read_wikiqa_data(f"data/dev_sampled.json")
    dev_set = dev_set[args.dev_slice:(args.num_dev)]
    
    prompt, _ = args.helper.prompt_for_verifying_answers(train_set, ['a', 'b'], 'verifying_question')
    print('prompt:')
    print(prompt)

    # finished consistency and processs
    predictions = read_full(args, consistency)
    new_predictions, cons = [], []
    for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
        ex = dev_set[i]
        con, new_p = consistency.post_process_consistency(ex, p, args)
        cons.append(con)
        new_predictions.append(new_p)
    predictions = new_predictions 
    [args.helper.post_process(p) for p in predictions] 

    verifying_qs = read_full(args, verifying_questions, slice = False)
    contexts = read_full(args, relevant_context, slice = False)
    
    if os.path.exists(result_cache_name(args)):
        verifying_answers = read_json(result_cache_name(args))
    else:
        print('running verifying answer generation')
        verifying_answers = []
        broken = False
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            id = ex['id']
            con = p['consistency']
            if con < args.consistency_threshold:
                sentences = rationale_tokenize(p['rationale'])
                vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                cont = [c for c in contexts if c['id']==id][0]['context'] 
                va = []
                for j, s in enumerate(sentences):
                    pars = cont[j][:3] #manually changing
                    try:
                        answer = in_context_manual_prediction(train_set, args.engine, args.helper, args.model, \
                            pars, vq[j])
                        va.append(answer['text'].lstrip())
                    except Exception as e:
                        print(f'EXCEPTION {e} ENCOUNTERED!')
                        print('pars: ', pars)
                        print('vq: ', vq[j])
                        answer = None
                        args.num_dev = i + args.dev_slice
                        broken = True
                        break
                # if not broken and not args.run_length_test:
                if not broken:
                    va = {'id': ex['id'], 'verifying_answers': va}
                    verifying_answers.append(va)
                elif broken:
                    print('ending early..')
                    break
        # save
        dump_json(verifying_answers, result_cache_name(args)) 
        print('saved to: ', result_cache_name(args))
    evaluate_manual_predictions(dev_set, predictions, verifying_qs, contexts, verifying_answers, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

