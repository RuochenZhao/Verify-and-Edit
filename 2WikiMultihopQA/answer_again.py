import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_wikiqa_data, f1auc_score, wiki_evaluation
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
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3)
    # for relevant_context retrieval
    parser.add_argument('--retrieval', choices=['wikipedia', 'drqa', 'google', 'dataset'])
    
    args = parser.parse_args()

    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/answering_again_tr{}-{}_dv{}-{}_thres{}_{}.json".format(\
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.consistency_threshold, args.retrieval)

def in_context_manual_prediction(ex, training_data, engine, prompt_helper, model, new_rationale, length_test_only):
    prompt, stop_signal = prompt_helper.prompt_for_answering_again(ex, training_data, new_rationale)
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
    logprob_records = []
    ansprob_records = []

    edited = 0
    edited_correctly = 0
    edited_falsely = 0
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        p_ans = pred['answer']
        id = ex['id']
        acc, (f1, pre, rec), gt_ans = wiki_evaluation(p_ans, ex["answer"])
        acc_records.append(acc)
        rat_acc = False
        rat_records.append(rat_acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        logprob_records.append(pred['joint_lobprob'])
        ansprob_records.append(pred['answer_logprob'])
        if do_print:
            if pred['consistency'] < args.consistency_threshold:
                oa = pred['original_answer']
                acc_before = oa == ex["answer"]
                print("--------------{} EX {} CONS {:.2f}--------------".format(ex['id'], acc, pred['consistency']))
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
                print('P RAT:', pred['new_rationale'])
                print('P:', p_ans, 'G:', ex['answer'])
                if acc_before and (not acc):
                    edited_falsely += 1
                elif (not acc_before) and acc:
                    edited_correctly += 1
    print(f'EDITED {edited} OUT OF {len(predictions)}')
    print(f'Edited {edited_correctly} correctly and {edited_falsely} falsely')
    mean_of_array = lambda x: sum(x) / len(x)
    print("EX", mean_of_array(acc_records), "RAT", mean_of_array(rat_records))
    print("F1: {:.2f}".format(mean_of_array(f1_records)), 
            "PR: {:.2f}".format(mean_of_array(pre_records)),
            "RE: {:.2f}".format(mean_of_array(rec_records)))
    print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
            ansprob_records, acc_records)))
    
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
    train_set = read_wikiqa_data(f"data/train_subset.json", manual_annotation_style=args.style)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    print('len(train_set): ', len(train_set))
    dev_set = read_wikiqa_data(f"data/dev_sampled.json")
    dev_set = dev_set[args.dev_slice:(args.num_dev)]
    
    prompt, _ = args.helper.prompt_for_answering_again(dev_set[0], train_set, 'new_rationale')
    print('prompt: ', prompt)

    contexts = read_full(args, relevant_context, slice = False)
    verifying_qs = read_full(args, verifying_questions, slice = False)
    verifying_as = read_full(args, verifying_answers, slice = False)

    if os.path.exists(result_cache_name(args)):
        print('reading from ', result_cache_name(args))
        predictions = read_json(result_cache_name(args))
    else:
        predictions = read_full(args, consistency)
        new_predictions, cons = [], []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            con, new_p = consistency.post_process_consistency(ex, p, args)
            cons.append(con)
            new_predictions.append(new_p)
        predictions = new_predictions 
        [args.helper.post_process(p) for p in predictions] 
        
        print('answering again generation')
        # change the original predictions
        lengths = []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            ex = dev_set[i]
            id = ex['id']
            con = p['consistency']
            if con < args.consistency_threshold: # only predict again when lower consistency
                va = [c for c in verifying_as if c['id']==id][0]['verifying_answers']
                new_rationale = rationale(va)
                new_p = in_context_manual_prediction(ex, train_set, args.engine, args.helper, args.model, \
                    new_rationale, length_test_only = args.run_length_test)
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
                    break
        # save
        dump_json(predictions, result_cache_name(args)) 
    [args.helper.post_process(p, change_rationale = False) for p in predictions]
    evaluate_manual_predictions(dev_set, verifying_qs, contexts, verifying_as, predictions, args, do_print=True)


if __name__=='__main__':
    args = _parse_args()
    test_few_shot_manual_prediction(args)
    print('finished')

