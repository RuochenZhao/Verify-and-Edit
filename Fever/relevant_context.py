import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_fever_data, read_jsonl
from comp_utils import model_embeddings, DPR_embeddings
import sklearn
from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import consistency
import pandas as pd
import torch
import verifying_questions
from prompt_helper import get_joint_prompt_helper
from matplotlib import pyplot as plt
import ngram
import numpy as np
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import time

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
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=3)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=300)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--consistency_threshold', type=float, default=3.6)
    parser.add_argument('--training_set',  default=False, action='store_true')
    parser.add_argument('--no_claim',  default=False, action='store_true')
    parser.add_argument('--plot_numbers',  default=False, action='store_true')
    parser.add_argument('--check_inclusion',  default=False, action='store_true')
    parser.add_argument('--retrieval', choices=['DPR', 'wikipedia', 'wikipedia_DPR', 'drqa', 'drqa_wiki', 'google'])
    parser.add_argument('--selection', choices=['tfidf', 'no_links'])
    parser.add_argument('--filtering', choices=['ngram', 'np_matching', 'no_filtering'])
    parser.add_argument('--drqa_path', type=str, default="misc/drqa_dev_300-default-pipeline.preds")
    parser.add_argument('--no_nei',  default=False, action='store_true')
    parser.add_argument('--use_sampled',  default=False, action='store_true')
    parser.add_argument('--google_path', type=str, default="misc/fever_google.pkl")
    
    args = parser.parse_args()

    specify_engine(args)
    args.helper = get_joint_prompt_helper(args.style)
    return args

def result_cache_name(args):
    ending = ''
    if args.filtering:
        ending = f'_{args.filtering}'
    if args.use_sampled:
        return "misc/relevant_context_sim_{}_tr{}-{}_dv{}-{}_{}_{}_sampled.json".format( \
                args.engine_name, args.train_slice, args.train_slice + args.num_shot, \
                args.dev_slice, args.num_dev, args.retrieval, args.selection)
    if args.training_set:
        ending += '_training_set'
    return "misc/relevant_context_sim_{}_tr{}-{}_dv{}-{}_{}_{}{}.json".format( \
            args.engine_name, args.train_slice, args.train_slice + args.num_shot, \
            args.dev_slice, args.num_dev, args.retrieval, args.selection, ending)

def evaluate_manual_predictions(dev_set, predictions, contexts, verifying_qs, do_print=False):
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        id = ex['id']
        if pred['consistency'] < args.consistency_threshold:
            cont = [c for c in contexts if c['id']==id][0]['context'] #list
            gt_cont = ex['pars']
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
    if not args.training_set:
        train_set = read_fever_data(f"data/train_subset.jsonl", manual_annotation_style=args.annotation)
        train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
        dev_set = read_jsonl("data/paper_test_processed.jsonl")
        if args.use_sampled:
            sampled_ids = read_json('data/sampled_ids.json')
            dev_set = [[d for d in dev_set if d['id']==id][0] for id in sampled_ids]
        dev_set = dev_set[args.dev_slice:args.num_dev]

        if args.use_sampled:
            predictions = read_full(args, consistency, slice = False)
            predictions = [[d for d in predictions if d['id']==id][0] for id in sampled_ids]
            predictions = predictions[args.dev_slice:args.num_dev]
        else:
            predictions = read_full(args, consistency)
        [args.helper.post_process(p) for p in predictions] 
        try:
            verifying_qs = read_full(args, verifying_questions, slice = False)
        except:
            verifying_qs = read_json(verifying_questions.result_cache_name(args))
    if os.path.exists(result_cache_name(args)):
        # finished all steps, evaluating
        contexts = read_json(result_cache_name(args))
    else:
        # finished consistency
        print('running context selection')
        contexts = []
        if args.retrieval == 'wikipedia_DPR':
            ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", model_max_length = 512)
            q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base", model_max_length = 512)
        else:
            embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
        
        if args.retrieval == 'DPR':
            encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
            searcher = FaissSearcher.from_prebuilt_index(
                'wikipedia-dpr-multi-bf',
                encoder
            )
            dataset = load_dataset("wiki_dpr", 'psgs_w100.multiset.compressed.no_embeddings')['train']
        if 'drqa' in args.retrieval:
            drqa_contexts = read_json_txt(args.drqa_path)
        if args.retrieval == 'google':
            with open(args.google_path, 'rb') as f:
                google_contexts = pickle.load(f)
                    
        broken = False
        includes_before = []
        includes_before_nums = []
        includes_after = []
        includes_after_nums = []
        numbers = []
        num_supps = []
        num_keyword_overlaps = []

        scores_before = []
        scores_after = []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            cont = []
            ex = dev_set[i]
            if args.training_set:
                p['consistency'] = 0
                ex['pars'] = ex['ground_truth_contexts']
            if p['consistency'] < args.consistency_threshold: # only predict again when lower consistency
                id = ex['id']
                if args.retrieval == 'google':
                    pars_old = [p for p in google_contexts if p['id']==str(id)][0]['qa_pair']
                if args.training_set:
                    sentences = [ex['question']]
                    vq = [ex['question']]
                else:
                    sentences = rationale_tokenize(p['rationale'])
                    vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                all_pars_text = []
                all_pars = []
                for j, s in enumerate(sentences):
                    # RETRIEVE
                    if args.retrieval in ['wikipedia', 'wikipedia_DPR']:
                        try:
                            pars_text = get_texts_to_rationale_wikipedia(vq[j], args.selection=="no_links")
                        except Exception as e:
                            print(f'Exception {e}')
                            args.num_dev = i + args.dev_slice
                            broken = True
                            break
                    elif args.retrieval == 'DPR':
                        # pars
                        hits = searcher.search(vq[j])
                        hits = [int(hits[i].docid) for i in range(5)]
                        pars_text = []
                        for h in hits:
                            pars_text += tokenize.sent_tokenize(dataset[h-1]['text'])
                    elif 'drqa' in args.retrieval:
                        if not args.training_set:
                            drqa_pred = [p for p in drqa_contexts if p['question']==vq[j]][0]
                        else:
                            try:
                                drqa_pred += [p for p in drqa_contexts if p['question']==ex['question']][0]
                            except:
                                print(ex['question'])
                                raise Exception('end')
                        pars_text = []
                        for par in drqa_pred['contexts']:
                            pars_text += tokenize.sent_tokenize(par)
                        if args.retrieval == 'drqa_wiki':
                            pars_text += get_texts_to_pages(drqa_pred['wiki_titles'], topk = 5)
                    elif args.retrieval == 'google':
                        pars_new = [p for p in pars_old if p['q']==vq[j]][0]['contexts']
                        pars_text = []
                        for p in pars_new:
                            if 'ab_snippet' in p:
                                pars_text.append(p['ab_snippet'])
                            else:
                                try:
                                    pars_text.append(p['snippet'])
                                except:
                                    print(p)
                                    raise Exception('end')
                    else:
                        pars_text = ex['pars']
                    pars_text = list(dict.fromkeys(pars_text)) #remove potential duplicates
                    all_pars_text += pars_text
                    if args.plot_numbers:
                        numbers.append(len(pars_text))
                        num_supps.append(len(ex['pars']))

                    if pars_text != []: # not empty list
                        if args.filtering == 'ngram':
                            pars = np.array([ngram.NGram.compare(vq[j], s, N=1) for s in pars_text])
                            # argsort: smallest to biggest
                            pars = pars.argsort()[::-1][:args.topk]
                        elif args.filtering != 'no_filtering':
                            if args.filtering == 'np_matching':
                                overlaps = check_keyword_overlap(vq[j], pars_text)
                                num_keyword_overlaps += overlaps
                                # filter only the ones >= 50%
                                pars_text = [pars_text[i] for i in range(len(pars_text)) if overlaps[i]>=0.3]
                            if args.retrieval == 'google':
                                # no need to rank, just take top k
                                pars = range(min(args.topk, len(pars_text)))
                            elif pars_text != []:
                                if args.selection=='tfidf':
                                    tfidf_vectorizer = TfidfVectorizer()
                                    tfidf_vectorizer = tfidf_vectorizer.fit([s] + pars_text)
                                    sen_embeds = tfidf_vectorizer.transform([s])
                                    par_embeds = tfidf_vectorizer.transform(pars_text)
                                elif args.retrieval== 'wikipedia_DPR':
                                    sen_embeds = [DPR_embeddings(q_encoder, q_tokenizer, vq[j], device)]
                                    par_embeds = [DPR_embeddings(ctx_encoder, ctx_tokenizer, s, device) for s in pars_text]
                                else:
                                    sen_embeds = [model_embeddings(s, embedding_model)]
                                    par_embeds = [model_embeddings(s, embedding_model) for s in pars_text]

                                pars = sklearn.metrics.pairwise.pairwise_distances(sen_embeds, par_embeds)
                                pars_before = pars[0]
                                scores_before += list(pars_before)
                                pars = pars.argsort(axis = 1)[0][:args.topk]
                                scores_after += [pars_before[i] for i in pars]
                    if pars_text == []: # empty list
                        print(f'EMPTY LIST ENCOUNTERED WITH {vq[j]}')
                        cont.append(['None'])
                    elif args.filtering != 'no_filtering':
                        try:
                            pars = [pars_text[i] for i in pars]
                        except:
                            print(id)
                            print('pars_text: ', pars_text)
                            raise Exception('end')
                        cont.append(pars)
                        all_pars += pars
                    else:
                        cont.append(pars_text)
                if not broken:
                    includes, total = check_inclu(ex['pars'], list(dict.fromkeys(all_pars_text)))
                    includes_before.append(includes/(total+1e-7)) #percentage
                    includes_before_nums.append((includes, total))
                    if args.filtering != 'no_filtering':
                        includes, total = check_inclu(ex['pars'], list(dict.fromkeys(all_pars)))
                        includes_after.append(includes/(total+1e-7)) #percentage
                        includes_after_nums.append((includes, total))
                    cont = {'id': ex['id'], 'context': cont, 'verifying_question': vq, \
                        'overall_question': ex['question'], 'ground_truth_contexts': ex['pars']}
                    contexts.append(cont)
                else:
                    break
        # save
        dump_json(contexts, result_cache_name(args)) 
        print('includes_before: ', str(sum(includes_before)/(len(includes_before)+1e-7)))
        print(f'scores with mean {np.mean(scores_before)} and std {np.std(scores_before)}')
        if args.filtering != 'no_filtering':
            print('includes_after: ', str(sum(includes_after)/(len(includes_after)+1e-7)))
            print(f'scores with mean {np.mean(scores_after)} and std {np.std(scores_after)}')
        print('includes_before_nums: ', includes_before_nums)
        print('includes_after_nums: ', includes_after_nums)
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
            plt.savefig(f"log/numbers_relevant_context_{args.selection}_{args.retrieval}_{args.filtering}.png")
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.hist(scores_after)
            ax1.set_title('scores of ranked sentences')
            ax2.hist(scores_before)
            ax2.set_title('scores of retrieved sentences')
            plt.savefig(f"log/scores_relevant_context_{args.selection}_{args.retrieval}_{args.filtering}.png")
        
    evaluate_manual_predictions(dev_set, predictions, contexts, verifying_qs, do_print=True)
    print(result_cache_name(args))
    

if __name__=='__main__':
    args = _parse_args()
    if args.retrieval == 'DPR':
        from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
    test_few_shot_manual_prediction(args)
    print('finished')

