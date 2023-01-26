import os
import argparse
from tqdm import tqdm
from utils import *
from dataset_utils import read_hotpot_data
from comp_utils import model_embeddings, DPR_embeddings
import sklearn
from manual_joint import post_process_manual_prediction_and_confidence
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import consistency
import pandas as pd
import torch
import verifying_questions
from matplotlib import pyplot as plt
from datasets import load_dataset
import pickle

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
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=6)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--topk', type=int, default=3) # can use 1000 for everything
    parser.add_argument('--show_result',  default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--with_context',  default=False, action='store_true')
    parser.add_argument('--DPR',  default=False, action='store_true')
    parser.add_argument('--wikipedia',  default=False, action='store_true')
    parser.add_argument('--wikipedia_DPR',  default=False, action='store_true')
    parser.add_argument('--drqa',  default=False, action='store_true')
    parser.add_argument('--google',  default=False, action='store_true')
    parser.add_argument('--consistency_threshold', type=float, default=3.5)
    parser.add_argument('--plot_numbers',  default=False, action='store_true')
    parser.add_argument('--check_inclusion',  default=False, action='store_true')
    parser.add_argument('--drqa_path', type=str, default="misc/hotpotqa-default-pipeline.preds")
    parser.add_argument('--google_path', type=str, default="misc/hotpotqa_google.pkl")
    parser.add_argument('--ablation',  default=False, action='store_true')
    
    args = parser.parse_args()
    if args.DPR + args.wikipedia + args.wikipedia_DPR >1:
        raise Exception('Cannot do DPR and wikipedia at the same time.')

    specify_engine(args)
    return args

def result_cache_name(args):
    ending = ''
    if args.DPR:
        ending += '_DPR'
    elif args.wikipedia:
        ending += '_wikipedia'
    elif args.drqa:
        ending += '_drqa'
    elif args.google:
        ending += '_google'
    if args.ablation:
        ending += '_ablation'
    return "misc/relevant_context_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}{}.json".format(args.annotation, \
            args.engine_name, args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
            args.num_distractor, args.style, ending)

def evaluate_manual_predictions(dev_set, dev_set_everything, predictions, contexts, model, style="p-e", do_print=False):
    for idx, (ex, ex_everything, pred) in enumerate(zip(dev_set, dev_set_everything, predictions)):
        id = ex['id']
        if pred['consistency'] < args.consistency_threshold:
            cont = [c for c in contexts if c['id']==id][0]['context'] #list
            gt_cont = [e['context'] for e in ex_everything['paragraphs'] if e['is_supp']]
            if do_print:
                print("--------------{} EX --------------".format(idx))
                print('question: ', ex['question'])
                sentences = rationale_tokenize(pred['rationale'])
                print('ground_truth_contexts: {}'.format(gt_cont))
                for j, (s, c) in enumerate(zip(sentences, cont)):
                    print('rationale_sentence {}: {}'.format(j, s))
                    print('relevant_contexts {}: {}'.format(j, c))

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set_everything = data = read_json("data/sim_dev.json")
    dev_set = dev_set[args.dev_slice:args.num_dev]
    dev_set_everything = dev_set_everything[args.dev_slice:args.num_dev]

    predictions = read_full(args, consistency)
    [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions] 
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
        contexts = read_json(result_cache_name(args))
    else:
        print('running context selection')
        contexts = []
        try:
            verifying_qs = read_json(verifying_questions.result_cache_name(args))
        except:
            verifying_qs = read_full(args, verifying_questions, slice = False)
       

        if args.wikipedia_DPR:
            ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", model_max_length = 512)
            q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base", model_max_length = 512)
        else:
            embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
        
        if args.DPR:
            encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
            searcher = FaissSearcher.from_prebuilt_index(
                'wikipedia-dpr-multi-bf',
                encoder
            )
            dataset = load_dataset("wiki_dpr", 'psgs_w100.multiset.compressed.no_embeddings')['train']
        if args.drqa:
            drqa_contexts = read_json_txt(args.drqa_path)
        if args.google:
            with open(args.google_path, 'rb') as f:
                google_contexts = pickle.load(f)
        broken = False
        includes_before = []
        includes_after = []
        numbers = []
        num_supps = []
        for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
            cont = []
            if p['consistency'] < args.consistency_threshold: # only predict again when lower consistency
                ex = dev_set[i]
                ex_everything = dev_set_everything[i]
                gt_cont = [e['context'] for e in ex_everything['paragraphs'] if e['is_supp']]
                sentences = rationale_tokenize(p['rationale'])
                id = ex['id']
                if args.google:
                    pars_old = [p for p in google_contexts if p['id']==id][0]['qa_pair']
                vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
                all_pars_text = []
                all_pars = []
                for j, s in enumerate(sentences):
                    # RETRIEVE
                    if args.wikipedia or args.wikipedia_DPR:
                        try:
                            pars_text = get_texts_to_rationale_wikipedia(vq[j])
                        except Exception as e:
                            print(f'Exception {e}')
                            args.num_dev = i + args.dev_slice
                            broken = True
                            break
                    elif args.DPR:
                        # pars
                        hits = searcher.search(vq[j])
                        hits = [int(hits[i].docid) for i in range(10)]
                        pars_text = []
                        for h in hits:
                            pars_text += tokenize.sent_tokenize(dataset[h-1]['text'])
                    elif args.drqa:
                        pars_old = [p for p in drqa_contexts if p['question']==vq[j]][0]['contexts']
                        pars_old += [p for p in drqa_contexts if p['question']==ex['question']][0]['contexts']
                        pars_text = []
                        for par in pars_old:
                            pars_text += tokenize.sent_tokenize(par)
                    elif args.google:
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
                        num_supps.append(len(gt_cont))
                    
                    # RANK
                    if pars_text != []: # not empty list
                        if args.google:
                            # no need to rank, just take top k
                            pars = range(args.topk)
                        else:
                            if args.wikipedia_DPR:
                                sen_embeds = [DPR_embeddings(q_encoder, q_tokenizer, vq[j], device)]
                                par_embeds = [DPR_embeddings(ctx_encoder, ctx_tokenizer, s, device) for s in pars_text]
                            else:
                                sen_embeds = [model_embeddings(vq[j], embedding_model)]
                                par_embeds = [model_embeddings(s, embedding_model) for s in pars_text]
                            pars = sklearn.metrics.pairwise.pairwise_distances(sen_embeds, par_embeds)
                            pars = pars.argsort(axis = 1)[0][:args.topk]
                        pars = [pars_text[i] for i in pars]
                        cont.append(pars)
                        all_pars += pars
                    else: # empty list
                        print(f'EMPTY LIST ENCOUNTERED WITH {vq[j]}')
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
            ax2.hist(num_supps)
            ax2.set_title('Numbers of ground truth supporting sentences')
            plt.savefig(f"log/numbers_relevant_context_drqa.png")
        
    evaluate_manual_predictions(dev_set, dev_set_everything, predictions, contexts, args.model, args.style, do_print=True)
    

if __name__=='__main__':
    args = _parse_args()
    if args.DPR:
        from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
    test_few_shot_manual_prediction(args)
    print('finished')

