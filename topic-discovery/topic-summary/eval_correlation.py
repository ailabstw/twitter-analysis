import sys
import re
import json
import argparse
import spacy
nlp = spacy.load('en_core_web_sm')
from typing import List
from collections import Counter
from tqdm import tqdm

from analysis_method import RecallEvaluator, PrecisionEvaluator, F1Evaluator
from util import read_df, read_json, read_stopwords, write_json

def sentence_to_feature(text, stopwords, lemmatized=True):
    if not lemmatized:
        text = ' '.join([token.lemma_ for token in nlp(text)])
        text = sentence_filter(text)
    
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
    
def article_to_feature(article:List[str], stopwords:set, lemmatized:bool=True):
    feature = set()
    for sentence in article:
        sentence_feature = sentence_to_feature(sentence, stopwords, lemmatized)
        feature.update(sentence_feature)
    return feature

def process_article(article):
    sentences = []
    for sent in article:
        sent = sentence_filter(sent)
        # filter empty token
        tokens = [t for t in sent.split(' ') if t != '']
        if len(tokens) > 0:
            sentences.append(sent)

    return sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path')
    parser.add_argument('--output_path')
    parser.add_argument('--candidate_path')
    parser.add_argument('--candidate_sentence')
    parser.add_argument('--stopword_path', default='dict/stopwords_en.txt')

    args = parser.parse_args()

    df = read_df(args.source_path)
    stopwords = read_stopwords(args.stopword_path)

    '''
        Process source data
    '''
    articles = list(df['lemma_content'])
    tweet_ids = list(df['tweet_id'])

    # convert source data to features
    article_features = [article_to_feature(article, stopwords) for article in tqdm(articles, desc='convert article to feature')]
    assert len(articles) == len(tweet_ids)
    
    '''
        Compute correlation score
    '''
    #evaluator = RecallEvaluator(article_features)
    f1_evaluator = F1Evaluator(article_features)
    recall_evaluator = RecallEvaluator(article_features)
    precision_evaluator = PrecisionEvaluator(article_features)
    if args.candidate_sentence:
        raise NotImplementedError
        # convert sentence to feature
        # Assume that candidate sentence is not lemmatized
        sentence_feature = sentence_to_feature(args.candidate_sentence, stopwords, lemmatized=False)
        result = evaluator(sentence_feature)

    elif args.candidate_path:
        candidates = read_json(args.candidate_path)
        for candidate in candidates:
            sentence = candidate['sentence'] 
            sentence_feature = sentence_to_feature(sentence, stopwords)
            f1 = f1_evaluator(sentence_feature)
            p = precision_evaluator(sentence_feature)
            r = recall_evaluator(sentence_feature)
            candidate['correlation_score'] = {'f1':f1, 'precision':p, 'recall':r}
    write_json(candidates, args.output_path)
