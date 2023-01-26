import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score
from comp_utils import safe_completion
from manual_joint import post_process_manual_prediction_and_confidence, post_process_manual_confidance
import consistency
import verifying_questions
import relevant_context
import verifying_answers

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
    parser.add_argument('--mode', choices=["simply_replace", "answer_with_context"])
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--DPR',  default=False, action='store_true')
    parser.add_argument('--wikipedia',  default=False, action='store_true')
    parser.add_argument('--wikipedia_DPR',  default=False, action='store_true')
    parser.add_argument('--drqa',  default=False, action='store_true')
    parser.add_argument('--google',  default=False, action='store_true')
    parser.add_argument('--ablation',  default=False, action='store_true')
    
    args = parser.parse_args()
    if args.DPR and args.wikipedia:
        raise Exception('Cannot do DPR and wikipedia at the same time.')
    specify_engine(args)
    return args

def result_cache_name(args):
    ending = ''
    if args.with_context:
        ending += '_with_context'
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
    return "misc/answering_again_{}_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_thres{}{}.json".format(args.annotation, \
        args.mode, args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.num_distractor, args.style, args.consistency_threshold, ending)

def in_context_manual_prediction(ex, training_data, engine, model, new_rationale, with_context, style="e-p", no_sep = False):
    prompt, stop_signal = prompt_for_manual_prediction(ex, training_data, style, with_context, rationale = new_rationale, no_sep = no_sep)
    if model == 'gpt3':
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
    rat_records = []
    f1_records, pre_records, rec_records = [], [], []
    ansprob_records = []

    edited = 0
    edited_correctly = []
    edited_incorrectly = []
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        
        p_ans = pred['answer']
        p_rat = pred['rationale']
        acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
        acc_records.append(acc)
        rat_acc = False
        rat_records.append(rat_acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        ansprob_records.append(pred['answer_logprob'])
        if do_print:
            print("--------------{} EX {} RAT {} F1 {:.2f} CONS {:.2f}--------------".format(idx, acc, rat_acc, f1, pred['consistency']))
            print('question: ', ex['question'])
            if pred['consistency'] < args.consistency_threshold:
                edited += 1
                id = ex['id']
                vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                cont = [c for c in contexts if c['id']==id][0]['context']
                va = [c for c in verifying_as if c['id']==id][0]['verifying_answers']
                print('original_rationale: ', pred['original_rationale'])
                print('original_answer: ', pred['original_answer'])
                sentences = rationale_tokenize(pred['original_rationale'])
                for j, (s, q, c, a) in enumerate(zip(sentences, vq, cont, va)):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, q))
                    print('verifying_answers {}: {}'.format(j, a))
                print('PR RAT: ', pred['new_rationale'])
                if acc:
                    edited_correctly.append(id)
                else:
                    edited_incorrectly.append(id)
            else:
                print('PR RAT:', p_rat)
            print('PR ANS:', p_ans)
            print('GT ANS:', gt_ans)
            print(json.dumps({'qas_id': ex['id'], 'answer': p_ans}))
    mean_of_array = lambda x: sum(x) / len(x)
    print(f'EDITED {edited} OUT OF {len(predictions)}')
    print("EX", mean_of_array(acc_records), "RAT", mean_of_array(rat_records))
    print("F1: {:.2f}".format(mean_of_array(f1_records)), 
            "PR: {:.2f}".format(mean_of_array(pre_records)),
            "RE: {:.2f}".format(mean_of_array(rec_records)))
    print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
            ansprob_records, acc_records)))
    print('edited correctly: ', edited_correctly)
    print('edited incorrectly: ', edited_incorrectly)

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
        r = r.lstrip()
        # decapitalize first letter
        r = r[0].upper() + r[1:]
        new_rationale.append(f"{sep} {r}")
    return ' '.join(new_rationale)

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:args.num_dev]

    print('')
    contexts = read_full(args, relevant_context, slice = False)
    try:
        verifying_as = read_full(args, verifying_answers, slice = False)
    except:
        if args.ablation:
            ori_cons = args.consistency_threshold
            args.consistency_threshold = 10
        verifying_as = read_json(verifying_answers.result_cache_name(args))
        print('reading from ', verifying_answers.result_cache_name(args))
        if args.ablation:
            args.consistency_threshold = ori_cons
    # finished verifying question generation
    verifying_qs = read_full(args, verifying_questions, slice = False)
    try:
        if args.ablation:
            args.ablation = False
            original_answer_again = read_json(result_cache_name(args))
            print('reading from ', result_cache_name(args))
            original_answer_again = [oa for oa in original_answer_again if oa['consistency']<args.consistency_threshold]
            ids = [rc['id'] for rc in original_answer_again]
            args.ablation = True
            dev_set = [p for p in dev_set if p['id'] not in ids]
            print('len(dev_set): ', len(dev_set))
            args.consistency_threshold = 10
        predictions = read_path(result_cache_name(args))
        print('cache: ', result_cache_name(args))
    except:
        # finished consistency
        predictions = read_full(args, consistency)
        [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions] 
        if args.ablation:
            args.ablation = False
            original_answer_again = read_json(result_cache_name(args))
            print('reading from ', result_cache_name(args))
            original_answer_again = [oa for oa in original_answer_again if oa['consistency']<args.consistency_threshold]
            ids = [rc['id'] for rc in original_answer_again]
            args.ablation = True
            predictions = [p for p in predictions if p['id'] not in ids]
            print('len(predictions): ', len(predictions))
            dev_set = [p for p in dev_set if p['id'] not in ids]
            print('len(dev_set): ', len(dev_set))
            args.consistency_threshold = 10
        
        print('answering again generation')
        # change the original predictions
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
                new_p = in_context_manual_prediction(ex, train_set, args.engine, args.model, new_rationale, args.with_context, style=args.style)
                # break
                if new_p!=None:
                    new_p['consistency'] = p['consistency']
                    new_p['original_rationale'] = p['rationale']
                    new_p['original_answer'] = p['answer']
                    new_p['new_rationale'] = new_rationale
                    new_p['answer'] = new_p['text']
                    new_p['id'] = id
                    predictions[i] = new_p
                else: #error, ending early
                    print('ENDING EARLY')
                    args.num_dev = i + args.dev_slice
                    dump_json(predictions, result_cache_name(args))
                    raise Exception('end')
        # save
        dump_json(predictions, result_cache_name(args)) 
    prompt, _ = prompt_for_manual_prediction(dev_set[0], train_set, args.style, args.with_context, rationale = '')
    print('PROMPT: ')
    print(prompt)
    print('\n\n\n')
    [post_process_manual_prediction_and_confidence(p, args.style, change_rationale = False) for p in predictions]
    evaluate_manual_predictions(dev_set, verifying_qs, contexts, verifying_as, predictions, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

