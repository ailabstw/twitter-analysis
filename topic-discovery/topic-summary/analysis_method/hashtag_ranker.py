from collections import Counter

class HashtagRanker():
    def __init__(self):
        pass

    def __call__(self, hashtags, topk=20):
        counter = Counter()
        counter.update(hashtags)

        return counter.most_common(topk)
