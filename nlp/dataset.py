import re
import pandas as pd
from time import time
from pathlib import Path
from .utils import preprocess

class Dataset:
  def __init__(self, filename, cleaned_col='cleaned',
               label_col='label', text_col='text'):
    self.filename = filename
    self.label_col = label_col
    self.text_col = text_col
    self.cleaned_col = cleaned_col

  @property
  def data(self):
    data = self.dataframe[[self.label_col, self.text_col]].copy()
    data.columns = ['label', 'text']
    return data

  @property
  def cleaned_data(self):
    data =  self.dataframe[[self.label_col, self.cleaned_col]]
    data.columns = ['label', 'text']
    return data

  def load(self):
    df = pd.read_csv(Path(self.filename).resolve())
    self.dataframe = df

  def preprocess_texts(self, quiet=False, stemming=False, no_emoji=False):
    self.dataframe['cleaned'] = preprocess(self.dataframe[self.text_col],
                                           quiet, stemming, no_emoji)

  def save(self, path):
    save_path = Path(path).resolve()
    self.dataframe.to_csv(save_path, index=None)
