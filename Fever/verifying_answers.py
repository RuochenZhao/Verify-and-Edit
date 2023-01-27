import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_fever_data
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
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=3)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3.6)
    parser.add_argument('--no_claim',  default=False, action='store_true')
    # for relevant_context retrieval
    parser.add_argument('--retrieval', choices=['DPR', 'wikipedia', 'wikipedia_DPR', 'drqa', 'drqa_wiki', 'google'])
    parser.add_argument('--selection', choices=['no_links', 'with_links', 'tfidf'])
    parser.add_argument('--filtering', choices=['ngram', 'np_matching', 'no_filtering'])
    parser.add_argument('--use_sampled',  default=False, action='store_true')
    parser.add_argument('--no_nei',  default=False, action='store_true')
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    
    args = parser.parse_args()

    args.helper = get_joint_prompt_helper(args.style)
    specify_engine(args)
    args.training_set = False #needed for context file 
    return args

def result_cache_name(args):
    end = ''
    if args.use_sampled:
        end = '_sampled'
    return "misc/verifying_answers_sim_{}_tr{}-{}_dv{}-{}_thres{}_{}_{}_{}{}.json".format( \
        args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.consistency_threshold, args.retrieval, args.selection, args.filtering, end)

def in_context_manual_prediction(engine, prompt_helper, model, pars, verifying_question, length_test_only):
    prompt, stop_signal = prompt_helper.prompt_for_verifying_answers(pars, verifying_question)
    if length_test_only:
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
    train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_fever_data(f"data/paper_test.jsonl")
    if args.use_sampled:
        sampled_ids = read_json('data/sampled_ids.json')
        dev_set = [[d for d in dev_set if d['id']==id][0] for id in sampled_ids]
        print('after filtering dev set: ', len(dev_set))
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    if args.show_prompt:
        prompt, _ = args.helper.prompt_for_verifying_answers(['a', 'b'], 'verifying_question')
        print(prompt)
        raise Exception('prompt shown')

    predictions = read_full(args, consistency, sampled = False, slice = False)
    [args.helper.post_process(p) for p in predictions] 
    print('args.use_sampled1: ', args.use_sampled)
    if args.use_sampled:
        print(len(predictions))
        predictions = [[d for d in predictions if d['id']==id][0] for id in sampled_ids]
        print('after filtering predictions: ', len(predictions))
    else:
        predictions = predictions[args.dev_slice:args.num_dev]
    print('args.use_sampled2: ', args.use_sampled)
    try:
        verifying_qs = read_full(args, verifying_questions, slice = False)
    except:
        verifying_qs = read_json(verifying_questions.result_cache_name(args))
    print('args.use_sampled3: ', args.use_sampled)
    try:
        contexts = read_full(args, relevant_context, slice = False)
    except:
        print('args.use_sampled: ', args.use_sampled)
        print('context path: ', relevant_context.result_cache_name(args))
        contexts = read_json(relevant_context.result_cache_name(args))
    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        verifying_answers = read_json(result_cache_name(args))
        ids = [v['id'] for v in verifying_answers]
        for id in ids:
            if id not in sampled_ids:
                print(id, ' not in !')
    else:
        print('running verifying answer generation')
        verifying_answers = []
        broken = False
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            id = ex['id']
            print(id)
            con = p['consistency']
            if con < args.consistency_threshold:
                sentences = rationale_tokenize(p['rationale'])
                vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                cont = [c for c in contexts if c['id']==id][0]['context'] 
                va = []
                for j, s in enumerate(sentences):
                    pars = cont[j][:3] #manually changing
                    try:
                        answer = in_context_manual_prediction(args.engine, args.helper, args.model, \
                            pars, vq[j], length_test_only = args.run_length_test)
                        if args.run_length_test:
                            verifying_answers.append(answer)
                        else:
                            va.append(answer['text'].lstrip())
                    except Exception as e:
                        print(f'EXCEPTION {e} ENCOUNTERED!')
                        print('pars: ', pars)
                        print('vq: ', vq[j])
                        answer = None
                        args.num_dev = i + args.dev_slice
                        broken = True
                        break
                if not broken and not args.run_length_test:
                    va = {'id': ex['id'], 'verifying_answers': va}
                    verifying_answers.append(va)
                elif broken:
                    print('ending early..')
                    break
        if not args.run_length_test:
            # save
            dump_json(verifying_answers, result_cache_name(args)) 
            print('saved to: ', result_cache_name(args))
        else:
            print(result_cache_name(args))
            print('MAX', max(verifying_answers), 'COMP', _MAX_TOKENS)
            print('AVG', sum(verifying_answers)/len(verifying_answers))
            print('TOTAL', sum(verifying_answers))
            print('$', sum(verifying_answers)/1000*0.02)
    evaluate_manual_predictions(dev_set, predictions, verifying_qs, contexts, verifying_answers, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

