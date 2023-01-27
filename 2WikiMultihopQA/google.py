import json
from serpapi import GoogleSearch
import pickle
from tqdm.notebook import tqdm
import os.path

key = 'replace_with_your_own'
dataset_name = '2wiki'

f = open('misc/verifying_questions_%s.json'%(dataset_name), 'rb') # make sure this file exists
data = json.load(f)


# query each question and save
for i in tqdm(range(len(data))):
    tmp_data = data[i]
    for j in range(len(tmp_data['verifying_questions'])):
        new_id = str(tmp_data['id']) + '_' + str(j)
        
        if os.path.isfile('%s_searchresults/'%(dataset_name)+new_id+'.pkl'):
            print(new_id, 'skipped...')
        else:
            tmp = tmp_data['verifying_questions'][j].strip()

            params = {
              "engine": "google",
              "q": tmp,
              "api_key": key
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            with open('%s_searchresults/'%(dataset_name)+new_id+'.pkl', 'wb') as f:
                pickle.dump(results, f)


# concatenate results
opt = []

for i in range(len(data)):
    tmp_data = data[i]
    final_opt = {'id': str(tmp_data['id']), 'qa_pair': []}
    
    for j in range(len(tmp_data['verifying_questions'])):
        tmp = tmp_data['verifying_questions'][j]
        tmp_id = str(tmp_data['id'])
        new_id = tmp_id + '_' + str(j)
        
        with open('%s_searchresults/'%(dataset_name)+new_id+'.pkl', 'rb') as f:
            tt = pickle.load(f)
            
        content = []
        if 'answer_box' in tt:
            if 'snippet' in tt['answer_box']:
                content.append({'ab_snippet': tt['answer_box']['snippet']})
        
        # organic answers
        if 'organic_results' in tt:
            for k in range(len(tt['organic_results'])):
                tmp_content = tt['organic_results'][k]

                if 'snippet' in tmp_content:
                    opt_content = {'snippet': tmp_content['snippet']}
                    if 'snippet_highlighted_words' in tmp_content:
                        opt_content['snippet_highlighted_words'] = tmp_content['snippet_highlighted_words']
                    content.append(opt_content)
        else:
            print(new_id)
            content.append('No information.')
                
        final_opt['qa_pair'].append({'q': tmp, 'contexts': content})
        
    opt.append(final_opt)

# save
with open('%s_google.pkl'%(dataset_name), 'wb') as f:
    pickle.dump(opt, f)