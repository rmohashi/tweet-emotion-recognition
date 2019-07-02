import re
import nltk
import pandas as pd
from time import time
from pathlib import Path

class Dataset:
  def __init__(self, filename, label_col='label', text_col='text'):
    nltk.download('stopwords')
    self.filename = filename
    self.label_col = label_col
    self.text_col = text_col

  @property
  def data(self):
    data = self.dataframe[[self.label_col, self.text_col]].copy()
    data.columns = ['label', 'text']
    return data

  @property
  def cleaned_data(self):
    data =  self.dataframe[[self.label_col, 'cleaned']]
    data.columns = ['label', 'text']
    return data

  def load(self):
    df = pd.read_csv(Path(self.filename).resolve())
    self.dataframe = df

  def preprocess(self):
    start = time()
    texts = self.dataframe[self.text_col].str.lower()
    texts = texts.str.replace(r"http\S+", "")
    texts = texts.str.replace(r"http", "")
    texts = texts.str.replace(r"@\S+", "")
    texts = texts.str.replace(r"[^a-z\']", " ")
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")
    texts = texts.apply(lambda x: ' '.join(x.split()))
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"n't", ' not')
    texts = texts.apply(self._remove_stop_words)
    self.dataframe['cleaned'] = texts
    print("Time to clean up: {:.2f}".format(time() - start))

  def _remove_stop_words(self, tweet):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    return ' '.join([word for word in tweet.split() if word not in stopwords])