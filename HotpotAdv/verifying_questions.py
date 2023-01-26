import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_hotpot_data
from comp_utils import safe_completion
from manual_joint import post_process_manual_prediction_and_confidence
import consistency

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
    parser.add_argument('--consistency_threshold', type=float, default=3.5)
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--ablation',  default=False, action='store_true')
    
    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    if args.ablation:
        return "misc/verifying_questions_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}.json".format(args.annotation, \
            args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
            args.num_distractor, args.style)
    else:
        return "misc/verifying_questions_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_thres{}.json".format(args.annotation, \
            args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
            args.num_distractor, args.style, args.consistency_threshold)

def in_context_manual_prediction(question, sentence, engine, model, style="p-e"):
    prompt, stop_signal = prompt_for_question_generation(question, sentence)
    if model == 'gpt3':
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = 1, temp=0.0, logprobs=5)        
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
            cont = [c for c in verifying_questions if c['id']==id][0]['verifying_questions']
            if do_print:
                print("--------------{} EX {} CONS--------------".format(idx, pred['consistency']))
                print('question: ', ex['question'])
                sentences = rationale_tokenize(pred['rationale'])

                for j, s in enumerate(sentences):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, cont[j]))

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:args.num_dev]

    # finished all steps, evaluating
    predictions = read_json(consistency.result_cache_name(args))
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions] 
    if args.ablation:
        args.ablation = False
        original_verifying_questions = read_json(result_cache_name(args))
        ids = [vq['id'] for vq in original_verifying_questions]
        args.ablation = True
        predictions = [p for p in predictions if p['id'] not in ids]
        print('len(predictions): ', len(predictions))
        dev_set = [p for p in dev_set if p['id'] not in ids]
        print('len(dev_set): ', len(dev_set))
        args.consistency_threshold = 10
    if os.path.exists(result_cache_name(args)):
        print('reading from ', result_cache_name(args))
        verifying_questions = read_json(result_cache_name(args))
    else:
        print('running verifying question generation')
        verifying_questions = []
        broken = False
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            con = p['consistency']
            if con < args.consistency_threshold and ex['id'] not in ids:
                vq = []
                sentences = rationale_tokenize(p['rationale'])
                for q, s in enumerate(sentences):
                    ################### STEP 2: VERIFY WITH Q ####################
                    question = in_context_manual_prediction(ex['question'], s, args.engine, args.model)['text'] 
                    if question != None:
                        vq.append(question)
                    else:
                        args.num_dev = len(verifying_questions) + args.dev_slice
                        broken = True
                        break
                if not broken:
                    vq = {'id': ex['id'], 'verifying_questions': vq}
                    verifying_questions.append(vq)
        # save
        dump_json(verifying_questions, result_cache_name(args)) 
        
    evaluate_manual_predictions(dev_set, predictions, verifying_questions, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

