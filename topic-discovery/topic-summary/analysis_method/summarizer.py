import networkx as nx
import math
from tqdm import tqdm

class SentenceGraph():
    def __init__(self, allow_selfloop=False):
        self.graph = nx.Graph()
        self.allow_selfloop = allow_selfloop

    def add_edge(self, n1, n2, weight=1):
        
        if not self.allow_selfloop and n1 == n2:
            # prevent self loop
            return
            
        if (n1, n2) in self.graph:
            self.graph.edges[(n1, n2)][weight] += weight
        else:
            self.graph.add_edge(n1, n2, weight=weight)

    def textrank(self):
        return nx.pagerank(self.graph)

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

class TextRankSummarizer():
    def __init__(self, min_len=None, max_len=None, allow_selfloop=False):
        # constraint the length of output sentences
        self.min_len = min_len
        self.max_len = max_len

        self.allow_selfloop = allow_selfloop

    def __call__(self, articles, topk=20):
        graph = SentenceGraph(allow_selfloop = self.allow_selfloop)

        # add in article edge weight
        for article in tqdm(articles, desc='add in article weight'):
            for i in range(len(article)-1):
                for j in range(i+1, len(article)):
                    assert i != j
                    graph.add_edge(article[i]['text'], article[j]['text'], weight=1)
        
        # add content similatity edge weight
        sentences = [sent for article in articles for sent in article]
        sent_num = len(sentences)
        def index_generator(l):
            for i in range(l-1):
                for j in range(i+1, l):
                    yield (i, j)
            return 

        def add_sentence_edge(index):
            i, j = index
            if sentences[i]['text'] != sentences[j]['text'] :
                sim = self.sentence_sim(sentences[i]['feature'], sentences[j]['feature'])
                if sim > 0 :
                    graph.add_edge(sentences[i]['text'], 
                                   sentences[j]['text'], 
                                   weight=sim)

        indices = index_generator(sent_num)
        for _ in tqdm(map(add_sentence_edge, indices), total=sent_num*(sent_num-1)//2):
            pass

        scores = graph.textrank()

        # TODO : MMR

        # only sort sentences with desired length
        sentences = []
        for s in scores.keys():
            token_num = len(s.split(' '))
            if (self.min_len is None or token_num>=self.min_len) and (self.max_len is None or token_num<=self.max_len):
                sentences.append(s)
        return sorted([(s, scores[s])for s in sentences], key=lambda x:x[1], reverse=True)[:topk]

    def sentence_sim(self, feature1, feature2):
        overlap = len(feature1 & feature2)
        if overlap == 0:
            return 0
        sim = overlap/(math.log(len(feature1)) + math.log(len(feature2)))
        return sim

if __name__ == '__main__':
    articles = [[{'text':'a b c', 'feature':set(['a', 'b', 'c'])}, {'text':'c d e', 'feature':set(['c', 'd', 'e'])}], 
                [{'text':'b c d', 'feature':set(['b', 'c', 'd'])}, {'text':'1 2 3', 'feature':set(['1', '2', '3'])}, {'text':'2 3 4', 'feature':set(['2', '3', '4'])}]]
    summarizer = TextRankSummarizer()
    print (summarizer(articles))

