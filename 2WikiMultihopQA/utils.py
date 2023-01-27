import pickle
import json
import subprocess
import regex as re
import spacy
from nltk import tokenize
import wikipedia
import wikipediaapi
import os
import copy
from textblob import TextBlob
import nltk

wiki_wiki = wikipediaapi.Wikipedia('en')
nlp = spacy.load("en_core_web_sm")
_ENT_TYPES = ['EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']

def run_cmd(cmds, is_pred_task=False):
    if is_pred_task:
        cmds.append('--run_pred')
    print("Running CMD")
    print(" ".join(cmds))
    try:
        output = subprocess.check_output(cmds, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)
        raise e
    output = output.decode()
    return output
    
PE_STYLE_SEP = " The reason is as follows."
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]

def convert_paragraphs_to_context(s, connction='\n', pars = None):
    if pars == None: # use s 
        return connction.join(['{}'.format(p) for i, p in enumerate(s['pars'])])
    else:
        return connction.join(['{}'.format(p) for i, p in enumerate(pars)])

def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)

def read_json_txt(fname):
    queries = []
    for line in open(fname):
        data = json.loads(line)
        queries.append(data)
    return queries

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def read_jsonlines(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [json.loads(x) for x in lines]

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def add_engine_argumenet(parser):
    parser.add_argument('--engine',
                            default='text-davinci-003',
                            choices=['davinci', 'text-davinci-001', 'text-davinci-002'])

def specify_engine(args):
    if args.model=='gptj':
        args.engine_name = 'gptj'
        args.engine = 'gptj'
    else:
        args.engine_name = args.engine

def get_sep_text(pred, style):
    if style == "e-p":
        for sep in EP_POSSIBLE_SEP_LIST:
            if sep in pred["text"]:
                return sep
        return None
    else:
        raise RuntimeError("Unsupported decoding style")

def prompt_for_manual_prediction(ex, shots, style, with_context=False, rationale = None, answer = False, \
                                pars = None, verifying_question = None):
    stop_signal = "\n\n"
    if style == "e-p":
        if answer: # for verifying answers only using 3-shot, no rationale, with context
            showcase_examples = [
            "{}\nQ: {}\nA: {}.{}".format(
                convert_paragraphs_to_context(s), s["question"], s["manual_answer"], stop_signal) for s in shots[:3]
            ]
        else:
            if with_context:
                showcase_examples = [
                    "{}\nQ: {}\nA: {}{} {}.{}".format(
                        convert_paragraphs_to_context(s), s["question"], s["manual_rationale"], EP_STYLE_SEP,
                        s["answer"], stop_signal) for s in shots
                ]
            else:
                showcase_examples = [
                    "Q: {}\nA: {}{} {}.{}".format(
                        s["question"], s["manual_rationale"], EP_STYLE_SEP,
                        s["answer"], stop_signal) for s in shots
                ]
        if answer:
            q = verifying_question
        else:
            q = ex["question"]

        end_question = "Q: {}\nA:".format(q)
        if rationale != None:
            end_question = "{} {}{}".format(end_question, rationale, EP_STYLE_SEP)
        if answer or with_context: # for verifying answers , with context
            input_example = "{}\n{}".format(convert_paragraphs_to_context(ex, pars = pars), end_question)
        else:
            input_example = end_question
        prompt = "".join(showcase_examples + [input_example])
    else:
        raise RuntimeError(f"Unsupported prompt style {style}")
    return prompt, stop_signal

def rationale_tokenize(sen):
    sents = re.split("(First, )|(Second, )|(Third, )|(Fourth, )", sen)
    invalid = ['First, ', 'Second, ', '', None]
    sents = [s for s in sents if s not in invalid]
    return sents

# wikipedia utils
def find_ents(rationale):
    doc = nlp(rationale)
    valid_ents = []
    for ent in doc.ents:
        if ent.label_ in _ENT_TYPES:
            valid_ents.append(ent.text)
    return valid_ents

def relevant_pages_to_ents(valid_ents, topk = 5):
    titles = []
    for ve in valid_ents:
        title = wikipedia.search(ve)[:topk]
        titles += title
    titles = list(dict.fromkeys(titles))
    return titles

def relevant_pages_to_rationale(rationale, topk = 5):
    return wikipedia.search(rationale)[:topk]

def get_linked_pages(wiki_pages, topk = 5):
    linked_ents = []
    for wp in wiki_pages:
        linked_ents += list(wp.links.values())
        if topk != -1:
            linked_ents = linked_ents[:topk]
    return linked_ents

def get_texts_to_pages(pages, topk = 2):
    texts = []
    for p in pages:
        text = p.text
        texts += tokenize.sent_tokenize(text)[:topk]
    return texts

def get_wiki_objs(pages):
    return [wiki_wiki.page(title) for title in pages]

def get_texts_to_rationale_wikipedia(rationale, no_links):
    valid_ents = find_ents(rationale)
    # find pages
    pages_to_ents = relevant_pages_to_ents(valid_ents, topk = 5)
    pages_to_rationale = relevant_pages_to_rationale(rationale, topk = 5)
    page_titles = pages_to_ents + pages_to_rationale
    pages = get_wiki_objs(page_titles)
    new_pages = []
    for i, pt in enumerate(page_titles):
        if 'disambiguation' in pt:
            new_pages += get_linked_pages([pages[i]], topk = -1)
        else:
            new_pages += [pages[i]]
    pages = new_pages
    if not no_links:
        # add linked pages
        pages + get_linked_pages(pages, topk = 5)
    # get rid of duplicates
    pages = list(dict.fromkeys(pages))
    # get texts
    texts = get_texts_to_pages(pages, topk = 5)
    return texts

def read_path(pred_path):
    if os.path.exists(pred_path):
        predictions = read_json(pred_path)
    else:
        raise Exception(f'file {pred_path} unfound')
    return predictions

def read_full(args, module, slice = True,  full_num = 1000):
    new_args = copy.deepcopy(args)
    new_args.dev_slice = 0
    new_args.num_dev = full_num
    predictions = read_path(module.result_cache_name(new_args))
    if slice:
        predictions = predictions[args.dev_slice:args.num_dev]
    return predictions

def check_inclu(gt_pars, pars_text):
    pars_text = [p.replace(' ', '') for p in pars_text]
    includes = []
    for par in gt_pars:
        p = par.replace(' ', '')
        inc = [((p in pt) or (pt in p)) for pt in pars_text]
        includes.append((sum(inc)>=1))
    return sum(includes), len(includes)

def get_keywords(rationale):
    tokenized = nltk.word_tokenize(rationale)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
    nps = TextBlob(rationale).noun_phrases
    nouns = [n for n in nouns if n not in ' '.join(nps)]
    ents = find_ents(rationale)
    result = nouns + nps + ents
    return result

def check_keyword_overlap(q, contexts):
    q_keywords = get_keywords(q)
    overlap = []
    for context in contexts:
        c_keywords = get_keywords(context)
        common = [value for value in q_keywords if value in c_keywords]
        overlap.append(len(common)/(len(q_keywords)+0.01))
    return overlap