import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_wikiqa_data
from comp_utils import model_embeddings
import sklearn
from sentence_transformers import SentenceTransformer
import consistency
import pandas as pd
import torch
import verifying_questions
from prompt_helper import get_joint_prompt_helper
from matplotlib import pyplot as plt
import numpy as np
from nltk import tokenize
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3)
    parser.add_argument('--training_set',  default=False, action='store_true')
    parser.add_argument('--no_claim',  default=False, action='store_true')
    parser.add_argument('--plot_numbers',  default=False, action='store_true')
    parser.add_argument('--check_inclusion',  default=False, action='store_true')
    parser.add_argument('--retrieval', choices=['wikipedia', 'drqa', 'google', 'dataset'])
    parser.add_argument('--drqa_path', type=str, default="misc/verifying_questions-default-pipeline.preds")
    parser.add_argument('--google_path', type=str, default="misc/2wiki_google.pkl")
    
    
    args = parser.parse_args()

    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    return "misc/relevant_context_tr{}-{}_dv{}-{}_{}_thres{}.json".format( \
        args.train_slice, args.train_slice + args.num_shot, \
        args.dev_slice, args.num_dev, args.retrieval, args.consistency_threshold)

def evaluate_manual_predictions(dev_set, predictions, contexts, verifying_qs, do_print=False):
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        id = ex['id']
        if pred['consistency'] < args.consistency_threshold:
            cont = [c for c in contexts if c['id']==id][0]['context'] #list
            gt_cont = ex['supp_pars']
            vqs = [c for c in verifying_qs if c['id']==id][0]['verifying_questions'] #list
            if do_print:
                print("--------------{} EX --------------".format(idx))
                print('question: ', ex['question'])
                sentences = rationale_tokenize(pred['rationale'])
                print('ground_truth_contexts: {}'.format(gt_cont))
                for j, (s, q, c) in enumerate(zip(sentences, vqs, cont)):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('verifying_question {}: {}'.format(j, q))
                    print('relevant_contexts {}: {}'.format(j, c))

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_wikiqa_data(f"data/train_subset.json", manual_annotation_style=args.style)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    print('len(train_set): ', len(train_set))
    dev_set = read_wikiqa_data(f"data/dev_sampled.json")
    dev_set = dev_set[args.dev_slice:(args.num_dev)]
    
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
    
    if os.path.exists(result_cache_name(args)):
        # finished all steps, evaluating
        contexts = read_json(result_cache_name(args))
    else:
        contexts = []
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
        
        if 'drqa' in args.retrieval:
            drqa_contexts = read_json_txt(args.drqa_path)
        if args.retrieval == 'google':
            with open(args.google_path, 'rb') as f:
                google_contexts = pickle.load(f)
        
        broken = False
        includes_before = []
        includes_after = []
        numbers = []
        num_supps = []
        num_keyword_overlaps = []

        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            cont = []
            if p['consistency'] < args.consistency_threshold: # only predict again when lower consistency
                ex = dev_set[i]
                gt_cont = ex['supp_pars']
                sentences = rationale_tokenize(p['rationale'])
                id = ex['id']
                if args.retrieval == 'google':
                    pars_old = [p for p in google_contexts if p['id']==str(id)][0]['qa_pair']
                vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                all_pars_text = []
                all_pars = []
                for j, s in enumerate(sentences):
                    # RETRIEVE
                    if args.retrieval in ['wikipedia']:
                        try:
                            pars_text = get_texts_to_rationale_wikipedia(vq[j], False)
                        except Exception as e:
                            print(f'Exception {e}')
                            args.num_dev = i + args.dev_slice
                            broken = True
                            break
                    elif 'drqa' in args.retrieval:
                        pars_old = [p for p in drqa_contexts if p['question']==vq[j]][0]['contexts']
                        pars_old += [p for p in drqa_contexts if p['question']==ex['question']][0]['contexts']
                        pars_text = []
                        for par in pars_old:
                            pars_text += tokenize.sent_tokenize(par)
                    elif args.retrieval == 'google':
                        pars_new = [p for p in pars_old if p['q']==vq[j]][0]['contexts']
                        pars_text = []
                        for p in pars_new:
                            if 'ab_snippet' in p:
                                pars_text.append(p['ab_snippet'])
                            elif 'snippet' in p:
                                pars_text.append(p['snippet'])
                    else:
                        pars_text = ex['all_pars']
                    pars_text = list(dict.fromkeys(pars_text)) #remove potential duplicates
                    all_pars_text += pars_text

                    if args.plot_numbers:
                        numbers.append(len(pars_text))
                        num_supps.append(len(gt_cont))

                    if pars_text != []: # not empty list
                        if args.retrieval == 'google':
                            # no need to rank, just take top k
                            if len(pars_text) < 3:
                                print(id)
                            pars = range(min(args.topk, len(pars_text)))
                        else:
                            sen_embeds = [model_embeddings(s, embedding_model)]
                            par_embeds = [model_embeddings(s, embedding_model) for s in pars_text]
                            
                            pars = sklearn.metrics.pairwise.pairwise_distances(sen_embeds, par_embeds)
                            pars = pars.argsort(axis = 1)[0][:args.topk]
                        pars = [pars_text[i] for i in pars]
                        cont.append(pars)
                        all_pars += pars
                    if pars_text == []: # empty list
                        logging.info(f'EMPTY LIST ENCOUNTERED WITH {vq[j]}')
                        cont.append(['None'])
                if not broken:
                    includes, total = check_inclu(gt_cont, list(dict.fromkeys(all_pars_text)))
                    includes_before.append(includes/(total+1e-7)) #percentage
                    includes, total = check_inclu(gt_cont, list(dict.fromkeys(all_pars)))
                    includes_after.append(includes/(total+1e-7)) #percentage
                    cont = {'id': ex['id'], 'context': cont}
                    contexts.append(cont)
                else:
                    break
        # save
        dump_json(contexts, result_cache_name(args)) 
        print('includes_before: ', str(sum(includes_before)/(len(includes_before)+0.01)))
        print('includes_after: ', str(sum(includes_after)/(len(includes_after)+0.01)))
        if args.plot_numbers:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.hist(numbers)
            ax1.set_title('Numbers of retrived sentences')
            if args.filtering == 'np_matching':
                ax2.hist(num_keyword_overlaps)
                ax2.set_title('Numbers of overlapping keywords')
            else:
                ax2.hist(num_supps)
                ax2.set_title('Numbers of ground truth supporting sentences')
            plt.savefig(f"log/numbers_relevant_context_{args.retrieval}.png")
            
    evaluate_manual_predictions(dev_set, predictions, contexts, verifying_qs, do_print=True)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    if args.retrieval == 'DPR':
        from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
    test_few_shot_manual_prediction(args)
    print('finished')
