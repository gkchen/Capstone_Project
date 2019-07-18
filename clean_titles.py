import pandas as pd
from random import sample
import dill
import re
from tqdm import tqdm

from sklearn import base
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

class TextPreProcess(base.BaseEstimator, base.TransformerMixin):
    """
    Input  : document list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text
    """

    def __init__(self, ignore):
        self.en_stop = set(stopwords.words('english')) # English stop words list
        self.tokenizer = RegexpTokenizer(r'[a-z]+&?[a-z]+')
        self.lemmatizer = WordNetLemmatizer()
        self.replace = ignore

    def _process(self, text):
        raw = text.lower()
        for key, val in self.replace.items():
            raw = re.sub(key, val, raw)
        tokens = self.tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in self.en_stop]
        lemma_tokens = [self.lemmatizer.lemmatize(i) for i in stopped_tokens]
        output = ' '.join(lemma_tokens)
        return output

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        for text in tqdm(X, desc='pre-process text'):
            output.append(self._process(text))
        return output
