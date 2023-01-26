import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_hotpot_data
from comp_utils import safe_completion
from manual_joint import post_process_manual_prediction_and_confidence
import consistency
import verifying_questions
import relevant_context

_MAX_TOKENS = 70

# PROMOT CONTROL
PE_STYLE_SEP = " The reason is as follows."
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]
QUESTION_PROMPT = 'Sentence: First, the Olympics were held in Sochi, Russia.\nQuestion: Where are the Olymics held?\n\nSentence: Second, the 1993 World Champion figure skater is Oksana Baiul.\nQuestion: Who is the 1993 World Champion figure skater?\n\nSentence: '
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
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3)
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--DPR',  default=False, action='store_true')
    parser.add_argument('--google',  default=False, action='store_true')
    parser.add_argument('--wikipedia',  default=False, action='store_true')
    parser.add_argument('--wikipedia_DPR',  default=False, action='store_true')
    parser.add_argument('--drqa',  default=False, action='store_true')
    parser.add_argument('--ablation',  default=False, action='store_true')
    parser.add_argument('--show_prompt',  default=False, action='store_true')
    
    args = parser.parse_args()
    if args.DPR and args.wikipedia:
        raise Exception('Cannot do DPR and wikipedia at the same time.')
    specify_engine(args)
    return args

def result_cache_name(args):
    ending = ''
    if args.DPR:
        ending += '_DPR'
    elif args.wikipedia:
        ending += '_wikipedia'
    elif args.wikipedia_DPR:
        ending += '_wikipedia_DPR'
    elif args.drqa:
        ending += '_drqa'
    elif args.google:
        ending += '_google'
    if args.ablation:
        ending += '_ablation'
    return "misc/verifying_answers_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_thres{}{}.json".format(args.annotation, \
        args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.num_distractor, args.style, args.consistency_threshold, ending)

def in_context_manual_prediction(ex, training_data, engine, model, pars, verifying_question, style="e-p"):
    prompt, stop_signal = prompt_for_manual_prediction(ex, training_data, style, \
        answer = True, pars = pars, verifying_question = verifying_question)
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
                print("--------------{} EX {} CONS--------------".format(idx, pred['consistency']))
                print('question: ', ex['question'])
                sentences = rationale_tokenize(pred['rationale'])
                for j, (s, q, c, a) in enumerate(zip(sentences, vq, cont, va)):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, q))
                    print('contexts {}: {}'.format(j, c))
                    print('verifying_answers {}: {}'.format(j, a))

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:args.num_dev]

    print('')
    try:
        verifying_qs = read_full(args, verifying_questions, slice = False)
    except:
        verifying_qs = read_json(verifying_questions.result_cache_name(args))
    try:
        contexts = read_full(args, relevant_context, slice = False)
    except:
        contexts = read_json(relevant_context.result_cache_name(args))
    predictions = read_json(consistency.result_cache_name(args))
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions] 

    if args.show_prompt:
        prompt, stop_signal = prompt_for_manual_prediction(predictions[0], train_set, args.style, \
            answer = True, pars = ['a','b','c'], verifying_question = '')
        print(prompt)
        raise Exception('prompt shown')
    if args.ablation:
        args.ablation = False
        original_relevant_contexts = read_json(result_cache_name(args))
        ids = [rc['id'] for rc in original_relevant_contexts]
        args.ablation = True
        predictions = [p for p in predictions if p['id'] not in ids]
        print('len(predictions): ', len(predictions))
        dev_set = [p for p in dev_set if p['id'] not in ids]
        print('len(dev_set): ', len(dev_set))
        args.consistency_threshold = 10

    if os.path.exists(result_cache_name(args)):
        # finished all steps, evaluating
        verifying_answers = read_json(result_cache_name(args))
    else:
        # finished verifying question generation
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
                cont = [c for c in contexts if c['id']==id][0]['context'][:3] #manually changing
                va = []
                for j, s in enumerate(sentences):
                    pars = cont[j]
                    try:
                        answer = in_context_manual_prediction(ex, train_set, args.engine, args.model, \
                            pars, vq[j])['text'].lstrip()
                    except:
                        print('EXCEPTION ENCOUNTERED!')
                        print(sentences)
                        print(vq)
                        print(j)
                        answer = None
                    if answer != None:
                        va.append(answer)
                    else:
                        args.num_dev = len(verifying_answers) + args.dev_slice
                        broken = True
                        break
                if not broken:
                    va = {'id': ex['id'], 'verifying_answers': va}
                    verifying_answers.append(va)
                else:
                    break
            if broken:
                break
        # save
        dump_json(verifying_answers, result_cache_name(args)) 
    evaluate_manual_predictions(dev_set, predictions, verifying_qs, contexts, verifying_answers, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

