'''
    Compute word importance from a bunch of articles.
'''
import os
import math
from collections import Counter

def get_ngram(tokens, n):
    ret = []
    for i in range(len(tokens)-n):
        ngram = tokens[i:i+n]
        assert len(ngram) == n

        ret.append(ngram)
    return ret

class SloganExtractor():
    def __init__(self, method='tfidf', accept_ngram=None):
        self.method = method
        assert accept_ngram is None or isinstance(accept_ngram, tuple) and len(accept_ngram) == 2
        self.accept_ngram = accept_ngram

    def __call__(self, articles, topk=100, min_df=1):
        article_num = len(articles)
        # compute df and tf
        tf = Counter()
        df = Counter()
        for article in articles:
            if self.accept_ngram:
                candidates = []
                for sent in article:
                    tokens = sent.split(' ')
                    for n in range(self.accept_ngram[0], self.accept_ngram[0]+1):
                        candidates.extend(map(lambda x:' '.join(x), get_ngram(tokens, n)))
            else:
                candidates = article

            df.update(set(candidates))
            tf.update(candidates)

        # compute idf
        idf = {}
        for sent in df:
            idf[sent] = math.log(article_num/df[sent])

        # compute tf-idf
        tfidf = {}
        for sent in tf:
            tfidf[sent] = tf[sent]*idf[sent]

        # 
        if self.method == 'tfidf':
            if self.accept_ngram:
                key_fn = lambda x : tfidf[x]*len(x.split())
            else:
                key_fn = lambda x : tfidf[x]
        elif self.method == 'df':
            if self.accept_ngram:
                key_fn = lambda x : df[x]*len(x.split())
            elif self.accept_ngram:
                key_fn = lambda x : df[x]

        topk_sents = []
        for sent in sorted(df.keys(), key=key_fn, reverse=True):
            if df[sent] < min_df:
                continue

            topk_sents.append({'sentence':sent,
                               'tfidf':tfidf[sent],
                               'tf':tf[sent],
                               'df':df[sent]}
                                )
            if topk_sents == topk:
                break

        return topk_sents
