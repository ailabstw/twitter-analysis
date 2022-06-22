import json
import pandas as pd
from tqdm import tqdm
from collections import Counter

def read_df(path):
    print (f'read data frame from {path}')
    if path.endswith('pkl'):
        return pd.read_pickle(path)
    elif path.endswith('csv'):
        return pd.read_csv(path)
    elif path.endswith('json'):
        return pd.read_json(path, orient="index", dtype=False)
    raise Exception(f'Unknown file type : {path}')

def write_df(df, path):
    print (f'write data frame to {path}')
    if path.endswith('pkl'):
        return df.to_pickle(path)
    elif path.endswith('csv'):
        return df.to_csv(path)
    elif path.endswith('json'):
        return df.to_json(path, orient="index", force_ascii=False, indent=1)
    raise Exception(f'Unknown file type : {path}')
 
def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data   

def write_json(data, path):
    print (f'write data to {path}')
    with open(path, 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

def read_stopwords(path):
    print (f'read stopwords from {path}')
    stopwords = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip(' \n')
            if line!= '':
                stopwords.append(line)
    return set(stopwords)


def sentence_to_feature(text, stopwords):
    def valid_token(token):
        if token == '':
            return False
        if token in stopwords:
            return False
        if token in ',%\'':
            return False
        if token == '<people>':
            return False
        return True
    feature = set([token for token in text.split(' ') if valid_token(token)])
    return feature

def build_word_dict(articles, stopwords):
    # comute dictionary
    word_counter = Counter()
    for article in tqdm(articles, desc='build word dict'):
        for sent in article:
            word_counter.update([token for token in sent.split(' ') if token not in stopwords])
    return word_counter.most_common(100)