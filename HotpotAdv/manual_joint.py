import os
import argparse

from tqdm import tqdm

from utils import *
from few_shot import convert_paragraphs_to_context
from dataset_utils import read_hotpot_data, hotpot_evaluation_with_multi_answers, f1auc_score, read_incorrect_answers
from comp_utils import safe_completion, length_of_prompt


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
    return args

def result_cache_name(args):
    return "misc/manual{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.json".format(args.annotation, args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.num_distractor, args.style)

# return prompt stop_signal
def prompt_for_manual_prediction(ex, shots, style, with_context):
    stop_signal = "\n\n"
    # E—P
    if style == "e-p":
        if with_context:
            showcase_examples = [
                "{}\nQ: {}\nA: {}{} {}.\n".format(
                    convert_paragraphs_to_context(s), s["question"], s["manual_rationale"], EP_STYLE_SEP,
                    s["answer"]) for s in shots
            ]
            input_example = "{}\nQ: {}\nA:".format(convert_paragraphs_to_context(ex), ex["question"])
        else:
            showcase_examples = [
                "Q: {}\nA: {}{} {}.\n".format(
                    s["question"], s["manual_rationale"], EP_STYLE_SEP,
                    s["answer"]) for s in shots
            ]
            input_example = "Q: {}\nA:".format(ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")
    return prompt, stop_signal    

def in_context_manual_prediction(with_context, ex, training_data, engine, style="p-e", length_test_only=False):
    prompt, stop_signal = prompt_for_manual_prediction(ex, training_data, style, with_context)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, temp=0.0, logprobs=5)        
        if pred != None:   
            pred["prompt"] = prompt    
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = "null"
            pred["completion_offset"] = len(prompt)
    return pred

def get_sep_text(pred, style):
    if style == "e-p":
        for sep in EP_POSSIBLE_SEP_LIST:
            if sep in pred["text"]:
                return sep
        return None
    else:
        raise RuntimeError("Unsupported decoding style")

def post_process_manual_prediction(p, style, change_rationale = True):
    text = p["text"]
    text = text.strip()

    # place holder
    answer = "null"
    rationale = "null"
    rationale_indices = []
    if style == "p-e":         
        sep = PE_STYLE_SEP
        if sep in text:
            segments = text.split(sep)   
            answer = segments[0].strip().strip('.')
            rationale = segments[1].strip()
    elif style == "e-p":
        sep = get_sep_text(p, style)
        if sep is not None:
            segments = text.split(sep)
            if len(segments) == 2:
                answer = segments[1].strip().strip('.')
                rationale = segments[0].strip()
            else:
                answer = segments[0].strip().strip('.')
        else:
            answer = text
    else:
        raise RuntimeError("Unsupported decoding style")
    
    p["answer"] = answer
    if change_rationale:
        p["rationale"] = rationale
        p["rationale_indices"] = rationale_indices
    return answer, rationale

def post_process_manual_confidance(pred, style):
    completion_offset = pred["completion_offset"]
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    # exclusive idxs
    if "<|endoftext|>" in tokens:
        completion_end_tok_idx = tokens.index("<|endoftext|>") + 1
    else:
        completion_end_tok_idx = len(tokens)
    completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]

    if style == "p-e":            
        if PE_STYLE_SEP in pred["text"]:
            sep_token_offset = completion_offset + pred["text"].index(PE_STYLE_SEP)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            ans_logprob = sum(completion_probs[:sep_start_idx - 1])
            rat_logprob = sum(completion_probs[(sep_start_idx + 6):])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0
    elif style == "e-p":
        sep_text = get_sep_text(pred, style)
        if sep_text is not None:            
            sep_token_offset = completion_offset + pred["text"].index(sep_text)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            rat_logprob = sum(completion_probs[:sep_start_idx + 3])
            ans_logprob = sum(completion_probs[(sep_start_idx + 3):-1])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0
    else:
        raise RuntimeError("Unsupported decoding style")

    pred["answer_logprob"] = ans_logprob
    pred["rationale_logprob"] = rat_logprob
    pred["joint_lobprob"] = ans_logprob + rat_logprob
    return ans_logprob, rat_logprob

def post_process_manual_prediction_and_confidence(pred, style, change_rationale = True):
    # process answer and rationale
    post_process_manual_prediction(pred, style, change_rationale = True)
    post_process_manual_confidance(pred, style)


def evaluate_manual_predictions(dev_set, predictions, style="p-e", do_print=False):
    acc_records = []
    rat_records = []
    f1_records, pre_records, rec_records = [], [], []
    logprob_records = []
    ansprob_records = []

    certified_incorrect_answers = read_incorrect_answers()
    print('prompt: ')
    print(predictions[0]['prompt'])
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
    
        gt_rat = ' '.join(ex['rationale'])
        p_ans = pred['answer']
        p_rat = pred['rationale']
        acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
        acc_records.append(acc)
        rat_acc = False
        rat_records.append(rat_acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        logprob_records.append(pred['joint_lobprob'])
        ansprob_records.append(pred['answer_logprob'])
        if do_print and not acc:
            if ex['id'] in certified_incorrect_answers and p_ans in certified_incorrect_answers[ex['id']]:
                continue
            print("--------------{} EX {} RAT {} F1 {:.2f}--------------".format(idx, acc, rat_acc, f1))
            print(convert_paragraphs_to_context(ex))
            print(ex['question'])

            print('\nRAW TEXT', '[' + pred['text'].strip() + ']')
            print('PR ANS:', p_ans)
            # print('PR RAT:', p_rat)
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
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]


    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        predictions = []    
        for i, x in enumerate(tqdm(dev_set, total=len(dev_set), desc="Predicting")):
            pred = in_context_manual_prediction(args.with_context, x, train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test)
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
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]
    # acc
    analyze_few_shot_manual_prediction(args)

def analyze_few_shot_manual_prediction(args):
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    predictions = read_json(result_cache_name(args))
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]

    evaluate_manual_predictions(dev_set, predictions, args.style, do_print=True)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_manual_prediction(args)
    else:
        analyze_few_shot_manual_prediction(args)