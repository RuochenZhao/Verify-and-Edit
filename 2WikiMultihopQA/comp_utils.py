import os
import openai
from transformers import GPT2TokenizerFast, AutoTokenizer, BloomTokenizerFast
import torch
import time
import numpy as np
import os

_MAX_TOKENS = 144
_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
_TOKENIZER_GPTJ = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
_TOKENIZER_BLOOM = BloomTokenizerFast.from_pretrained("bigscience/bloom")
GPT3_LENGTH_LIMIT = 2049
openai.api_key="replace_with_your_own"

def bloom_style_tokenize(x):
    return _TOKENIZER_BLOOM.decode(_TOKENIZER_BLOOM.encode(x))

def gpt_style_tokenize(x, model='gpt3'):
    if model == 'gpt3':
        return _TOKENIZER.tokenize(x)
    elif model == 'gptj':
        return torch.from_numpy(np.array(_TOKENIZER_GPTJ.encode(x))).unsqueeze(0)
    else:
        raise NotImplementedError(f'model {model} unimplemented')

def gptj_decode(x, length, stop):
    text = _TOKENIZER_GPTJ.decode(x[length:])
    text = text.split(stop)[0].replace('<|endoftext|>', '')
    return text

def length_of_prompt(prompt, model='gpt3'):
    if model == 'gpt3':
        return len(_TOKENIZER.tokenize(prompt)) + _MAX_TOKENS
    elif model == 'gptj':
        return len(_TOKENIZER_GPTJ.tokenize(prompt)) + _MAX_TOKENS
    else:
        raise NotImplementedError(f'model {model} unimplemented')

def safe_completion(engine, prompt, MAX_TOKENS, stop, temp=0.0, logprobs=5, n = 1, num_tries = 0):
    len_prompt_token = len(_TOKENIZER.tokenize(prompt))    
    if MAX_TOKENS + len_prompt_token >= GPT3_LENGTH_LIMIT:
        print("OVERFLOW", MAX_TOKENS + len_prompt_token)
        return {
            "text": "overflow"
        }
    if n>1:
        temp = 0.7
    try:
        resp = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=MAX_TOKENS, stop=stop,
            temperature=temp, logprobs=logprobs, echo=True, n = n)
    except Exception as e:
        print(f'Encountered Error {e}, trying for the {num_tries} time.')
        time.sleep(10)
        if num_tries >= 10:
            return None
        else:
            return safe_completion(engine, prompt, MAX_TOKENS, stop, temp, logprobs, \
                n, num_tries = num_tries + 1)
    if n>1:
        return resp
    else:
        return resp["choices"][0]

def get_text_offset(token_ids, text, length):
    offset = 0
    offsets = []
    tokens = []
    for t in token_ids[length:]:
        token = _TOKENIZER_GPTJ.decode([t])
        tokens.append(token)
        offsets.append(offset)
        offset += len(token)
    return offsets, tokens

def gptj_completion(prompt, stop, model, device, temp = 0.1, n=1):
    input_ids = gpt_style_tokenize(prompt, model='gptj').to(device)
    result = model.generate(input_ids,max_new_tokens = _MAX_TOKENS, do_sample = True, temperature = temp, \
        num_return_sequences=n, return_dict_in_generate = True, output_scores = True)
    # process it
    length = input_ids.shape[-1]
    choices = []
    scores = torch.stack(result['scores'], dim=1).softmax(-1).max(-1).values
    for (i, r) in enumerate(result['sequences']):
        text = gptj_decode(r, length, stop)
        text_offset, tokens = get_text_offset(r, text, length)
        token_logprobs = scores[i].cpu().detach().numpy().tolist()
        token_ids = r.cpu().detach().numpy().tolist()
        logprobs = {'token_logprobs': token_logprobs, 'token_ids': token_ids, \
            'tokens': tokens, 'text_offset': text_offset}
        choices.append({'text': text, 'logprobs': logprobs})
    if n == 1:
        return choices[0], length
    else:
        return {'choices': choices, 'usage': {'prompt_tokens': 0}}, length

import json
import requests
API_TOKEN = "replace_with_your_own"
model_id = "bigscience/bloom"
# model_id = "gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    try:
        return json.loads(response.content.decode("utf-8"))
    except:
        return [{'generated_text': "overflow"}]

def get_hf_output(prompt, model): 
    len_prompt_token = len(_TOKENIZER.tokenize(prompt))
    if _MAX_TOKENS + len_prompt_token >= GPT3_LENGTH_LIMIT:
        print("OVERFLOW", _MAX_TOKENS + len_prompt_token)
        return {
            "text": "overflow"
        }
    data = query(
    {
        "inputs": str(prompt),
        "parameters": {"return_full_text": False,
        "max_new_tokens": 100,
        "max_time": 120.0,
        "output_scores": True,}
    }
    )
    try:
        text = data[0]['generated_text']
        return {'text': text}
    except:
        print('ending with: ', data)
        time.sleep(70)
        print(f'...Woke up!')
        data = query(
        {
            "inputs": str(prompt),
            "parameters": {"return_full_text": False,
            "max_new_tokens": 100,
            "max_time": 120.0,
            "output_scores": True,}
        }
        )
        text = data[0]['generated_text']
        return {'text': text}

def model_embeddings(sentence, model):
    embedding = model.encode([sentence])
    return embedding[0] #should return an array of shape 384

def DPR_embeddings(q_encoder, q_tokenizer, question, device):
    question_embedding = q_tokenizer(question, return_tensors="pt",max_length=5, truncation=True)
    with torch.no_grad():
        try:
            question_embedding = q_encoder(**question_embedding)[0][0]
        except:
            print(question)
            print(question_embedding['input_ids'].size())
            raise Exception('end')
    question_embedding = question_embedding.numpy()
    return question_embedding

def safe_embeddings(sentence):
    try:
        embedding = openai.Embedding.create(input = [sentence], model = 'text-similarity-babbage-001')['data'][0]['embedding']
    except Exception as e:
        print(f'Sleeping due to {e}...')
        time.sleep(70)
        print(f'...Woke up!')
        embedding = openai.Embedding.create(input = [sentence], model = 'text-similarity-babbage-001')['data'][0]['embedding']
    return embedding


def conditional_strip_prompt_prefix(x, p):
    if x.startswith(p):
        x = x[len(p):]
    return x.strip()
