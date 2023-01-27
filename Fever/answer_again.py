import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_fever_data, read_jsonl
from comp_utils import safe_completion, length_of_prompt
from prompt_helper import get_joint_prompt_helper, normalize_prediction
import consistency
import verifying_questions
import relevant_context
import verifying_answers

_MAX_TOKENS = 70

# PROMOT CONTROL
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]
RATIONALE_SEP_LIST = ["First,", "Second,", "Third,", "Fourth,", "Moreover,"]

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
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3.6)
    parser.add_argument('--mode', choices=["simply_replace", "answer_with_context"])
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--no_claim',  default=False, action='store_true')
    # for relevant_context retrieval
    parser.add_argument('--retrieval', choices=['DPR', 'wikipedia', 'wikipedia_DPR', 'drqa', 'drqa_wiki', 'google'])
    parser.add_argument('--selection', choices=['no_links', 'with_links', 'tfidf'])
    parser.add_argument('--filtering', choices=['ngram', 'np_matching', 'no_filtering'])
    parser.add_argument('--no_nei',  default=False, action='store_true')
    parser.add_argument('--use_sampled',  default=False, action='store_true')
    
    args = parser.parse_args()

    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    args.training_set = False
    return args

def result_cache_name(args):
    ending = ''
    if args.no_nei:
        ending = '_no_nei'
    if args.use_sampled:
        ending = '_sampled'
    return "misc/answering_again_{}_sim_{}_tr{}-{}_dv{}-{}_thres{}_{}_{}_{}{}.json".format(\
        args.mode, args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.consistency_threshold, args.retrieval, args.selection, args.filtering, ending )

def in_context_manual_prediction(ex, training_data, engine, prompt_helper, model, new_rationale, length_test_only):
    prompt, stop_signal = prompt_helper.prompt_for_answering_again(ex, training_data, new_rationale)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        return pred
    elif model == 'gpt3':
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = 1, temp=0.0, logprobs=5) 
        if pred != None:       
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = "null"
            pred["completion_offset"] = len(prompt)
    return pred

def evaluate_manual_predictions(dev_set, verifying_qs, contexts, verifying_as, predictions, args, do_print=False):
    acc_records = []
    all_probs = []
    all_texts = []

    edited = 0
    result_dict = {}
    edited_correctly = 0
    edited_falsely = 0
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        gt = ex["label"]
        id = ex['id']

        p_ans = normalize_prediction(pred['answer'])
        all_texts.append(p_ans)

        acc = p_ans == gt        
        acc_records.append(acc)    
        all_probs.append(pred['answer_logprob'])
        if do_print:
            if pred['consistency'] < args.consistency_threshold:
                oa = pred['original_answer']
                acc_before = oa == gt
                print("--------------{} EX {} CONS {:.2f}--------------".format(id, acc, pred['consistency']))
                print('question: ', ex['question'])
                edited += 1
                vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                cont = [c for c in contexts if c['id']==id][0]['context']
                va = [c for c in verifying_as if c['id']==id][0]['verifying_answers']
                try:
                    print('original_rationale: ', pred['original_rationale'])
                except:
                    print('original_rationale: ', 'none')
                print('original_answer: ', oa)
                sentences = rationale_tokenize(pred['original_rationale'])
                for j, (s, q, c, a) in enumerate(zip(sentences, vq, cont, va)):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, q))
                    print('contexts {}: {}'.format(j, c))
                    print('verifying_answers {}: {}'.format(j, a))
                print('P RAT:', pred['rationale'])
                print('P:', p_ans, 'G:', gt)
                if not acc:
                    k = f'{oa}_to_{p_ans}_withgt_{gt}'
                    print('k: ', k)
                    if k in result_dict:
                        result_dict[k] += 1
                    else:
                        result_dict[k] = 1
                if acc_before and (not acc):
                    edited_falsely += 1
                elif (not acc_before) and acc:
                    edited_correctly += 1
    print('results: ')
    for i in result_dict:
        print(i, ': ', result_dict[i])
    print(result_dict)
    print(f'EDITED {edited} OUT OF {len(predictions)}')
    print(f'Edited {edited_correctly} correctly and {edited_falsely} falsely')
    print(f'{sum(acc_records)} correct out of {len(acc_records)}')
    print("ACC", sum(acc_records) / len(acc_records))

def read_path(pred_path):
    if os.path.exists(pred_path):
        predictions = read_json(pred_path)
    else:
        raise Exception(f'file {pred_path} unfound')
    return predictions

def rationale(li):
    new_rationale = []
    for (i, r) in enumerate(li):
        if i < len(RATIONALE_SEP_LIST):
            sep = RATIONALE_SEP_LIST[i]
        else:
            sep = RATIONALE_SEP_LIST[-1]
        # remove leading whitespaces
        r = r.lstrip().replace('Yes, ', '').replace('No, ', '')
        # decapitalize first letter
        r = r[0].upper() + r[1:]
        new_rationale.append(f"{sep} {r}")
    return ' '.join(new_rationale)

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_jsonl(f"data/paper_test_processed.jsonl")
    if args.no_nei:
        dev_set = [d for d in dev_set if d['label'] != 'NOT ENOUGH INFO']
    if args.use_sampled:
        sampled_ids = read_json('data/sampled_ids.json')
        dev_set = [[d for d in dev_set if d['id']==id][0] for id in sampled_ids]
    dev_set = dev_set[args.dev_slice:args.num_dev]

    ori_ct = args.consistency_threshold
    args.consistency_threshold = 3.6
    try:
        contexts = read_full(args, relevant_context, slice = False)
    except:
        try:
            contexts = read_json(relevant_context.result_cache_name(args))
        except:
            contexts = read_full(args, relevant_context, slice = False, sampled = False)
    try:
        verifying_qs = read_full(args, verifying_questions, slice = False)
    except:
        verifying_qs = read_json(verifying_questions.result_cache_name(args))
    try:
        verifying_as = read_full(args, verifying_answers, slice = False)
    except:
        try:
            verifying_as = read_json(verifying_answers.result_cache_name(args))
            print('reading from ', verifying_answers.result_cache_name(args))
        except:
            verifying_as = read_full(args, verifying_answers, slice = False, sampled = False)

    args.consistency_threshold = ori_ct
    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_path(result_cache_name(args))
        [args.helper.post_process(p) for p in predictions] 
        if args.use_sampled:
            predictions = [[d for d in predictions if d['id']==id][0] for id in sampled_ids]
    else:
        predictions = read_full(args, consistency, sampled = False, slice = False)
        [args.helper.post_process(p) for p in predictions] 
        if args.use_sampled:
            predictions = [[d for d in predictions if d['id']==id][0] for id in sampled_ids]
        
        print('answering again generation')
        # change the original predictions
        lengths = []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            id = ex['id']
            con = p['consistency']
            if con < args.consistency_threshold: # only predict again when lower consistency
                if args.mode == 'simply_replace':
                    cont = [c for c in contexts if c['id']==id][0]['context']
                    cont = [c[0] for c in cont] # only use the first one
                    new_rationale = rationale(cont)
                elif args.mode == 'answer_with_context':
                    va = [c for c in verifying_as if c['id']==id][0]['verifying_answers']
                    new_rationale = rationale(va)
                new_p = in_context_manual_prediction(ex, train_set, args.engine, args.helper, args.model, \
                    new_rationale, length_test_only = args.run_length_test)
                if args.run_length_test:
                    lengths.append(new_p)
                elif new_p!=None:
                    new_p['consistency'] = p['consistency']
                    new_p['original_rationale'] = p['rationale']
                    new_p['original_answer'] = p['answer']
                    new_p['new_rationale'] = new_rationale
                    new_p['answer'] = new_p['text']
                    predictions[i] = new_p
                else: #error, ending early
                    print('ENDING EARLY')
                    args.num_dev = i + args.dev_slice
                    break
        if not args.run_length_test:
            # save
            dump_json(predictions, result_cache_name(args)) 
            [args.helper.post_process(p) for p in predictions]
        else:
            print(result_cache_name(args))
            print('MAX', max(lengths), 'COMP', _MAX_TOKENS)
            print('AVG', sum(lengths)/len(lengths))
            print('TOTAL', sum(lengths))
            print('$', sum(lengths)/1000*0.02)
    evaluate_manual_predictions(dev_set, verifying_qs, contexts, verifying_as, predictions, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

