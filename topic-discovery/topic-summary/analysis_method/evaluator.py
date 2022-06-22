import numpy as np
from typing import List, Set


def precision(hyp, ref):
    overlap = hyp & ref
    if len(overlap) == 0:
        return 0
    return len(overlap)/len(hyp)


def recall(hyp, ref):
    overlap = hyp & ref
    if len(overlap) == 0:
        return 0
    return len(overlap)/len(ref)

def f1(hyp, ref):
    r = recall(hyp, ref)
    p = precision(hyp, ref)
    if r == 0 or p == 0:
        return 0
    return 2*r*p/(r+p)

class RecallEvaluator():
    def __init__(self, source_data:List[Set[str]]):
        self.source_data = source_data

    def __call__(self, ref:Set[str], return_all:bool=False):
        recalls = []
        for hyp in self.source_data:
            score = recall(hyp, ref)
            recalls.append(score)

        if return_all:
            return recalls
        return np.mean(recalls)
        
class PrecisionEvaluator():
    def __init__(self, source_data:List[Set[str]]):
        self.source_data = source_data

    def __call__(self, ref:Set[str], return_all:bool=False):
        precisions = []
        for hyp in self.source_data:
            score = precision(hyp, ref)
            precisions.append(score)

        if return_all:
            return precisions
        return np.mean(precisions)
        
class F1Evaluator():
    def __init__(self, source_data:List[Set[str]]):
        self.source_data = source_data

    def __call__(self, ref:Set[str], return_all:bool=False):
        f1s = []
        for hyp in self.source_data:
            score = f1(hyp, ref)
            f1s.append(score)

        if return_all:
            return f1s
        return np.mean(f1s)
