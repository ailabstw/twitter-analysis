import re
import string

class TwitterSentenceFilter():
    def __init__(self, stopwords):
        self.stopword_pattern = '|'.join(map(lambda x:f'({re.escape(x)})', stopwords))

    def __call__(self, text, verbose=False):
        orig = text

        # remove special tokens in text
        text = self._remove_special_token(text)
        if len(text) == 0:
            return ''
        
        # normalize @name to <people>
        text = self.name_norm(text)
        
        # meanningful content should contain more than punctuations, stopwords, numbers, hashtags, <people> 
        if not self.has_meanningful_information(text):
            return ''

        text = re.sub('# (?=[^\s]+)', '#', text)
        text = re.sub('[\"!?:\(\)\.\-_\*\&\|\[\]]', ' ', text)
        
        # remove leading <people>
        text = re.sub('<people>(\s*,*\s*<people>)+', ' <people> ', text)
        text = re.sub('^\s*<people>', '', text)
        text = self.process_space(text)

        assert not re.match('^<people>', text), \
            f'make sure no leading <people>, but see \'{text}\'. \n orogin : {orig}'

        if self.is_useless_sentence(text):
            return '' 

        return text

    def name_norm(self, text):
        text = re.sub('@[^\s$]+(?=\s|$)', '<people>', text)
        text = re.sub('<people>(\s*<people>)+', ' <people> ', text)
        return text

    def process_space(self, text):
        text = re.sub('\s+', ' ', text)
        text = text.strip(', ')
        return text

    def has_meanningful_information(self, text):
        # remove hashtags
        text = self._remove_hashtag(text)

        # remove <people>
        text = self._remove_people(text)

        # remove number
        text = self._remove_number(text)

        # remove stopwords
        # because stopword may contain punctuation (like 's, n't),
        #   removing them before removing punctuations
        text = self._remove_stopwords(text)

        # remove punctuation
        text = self._remove_punctuation(text)

        # process space
        text = self.process_space(text)
        return text != ''

    def _remove_special_token(self, text):
        text = re.sub('<url>', ' ', text)
        text = re.sub('<phone>', ' ', text)
        text = re.sub('<email>', ' ', text)
        return text

    def _remove_hashtag(self, text):
        text = re.sub('# onev1 z21', ' ', text)
        text = re.sub('# [^#\s$]+(\s|$)', ' ', text)
        return text

    def _remove_people(self, text):
        return re.sub('(via )?<people>', ' ', text)

    def _remove_number(self, text):
        text = re.sub('\d+ \.', '', text)
        text = re.sub('\d+/\d+', '', text)
        return text

    def _remove_punctuation(self, text):
        return re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)

    def _remove_stopwords(self, text):
        return re.sub(self.stopword_pattern, ' ', text)

    def is_useless_sentence(self, text):
        useless_sentences = ['thank you', 'read more', 'learn more', 'follow my lists','do n\'t forget to subscribe', 'breaking news', 'follow my lists for more news', 'pass it on', 'see more on', 
                             'have you watched this latest video', 'new blog post']
        for sent in useless_sentences:
            if re.match(f'{sent}( .)?', text):
                return True
        return False
    
        
    def only_punc(self, text):
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = self.process_space(text)
        return len(text) == 0

    def only_hashtag(self, text):
        text = re.sub('# onev1 z21', ' ', text)
        text = re.sub('# [^#\s$]+(\s|$)', ' ', text)
        text = self.process_space(text)
        return len(text) == 0 or self.only_punc(text)

    def only_tag_people(self, text):
        text = re.sub('(via )?<people>', ' ', text)
        text = self.process_space(text)
        return len(text) == 0 or self.only_punc(text)

    def only_number(self, text):
        text = re.sub('\d+ \.', '', text)
        text = re.sub('\d+/\d+', '', text)
        text = self.process_space(text)
        return len(text) == 0 or self.only_punc(text)

    def only_stopwords(self, text):
        text = self.remove_stopwords(text)
        text = self.process_space(text)
        return len(text) == 0 or self.only_punc(text)


