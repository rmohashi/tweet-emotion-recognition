import os
import pickle
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from .dataset import Dataset

def tokenize(dataset_path,
             save_path,
             label_col='label',
             text_col='text',
             num_words=10000):
  dataset = Dataset(dataset_path, label_col=label_col, text_col=text_col)
  dataset.load()
  dataset.preprocess_texts()

  tokenizer = Tokenizer(num_words=num_words, lower=True)
  tokenizer.fit_on_texts(dataset.cleaned_data.text)

  file_to_save = Path(os.path.join(save_path, 'tokenizer.pickle')).resolve()
  with file_to_save.open('wb') as file:
    pickle.dump(tokenizer, file)

  print('Saved under: "' + file_to_save.as_posix() + '"')

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('dataset_path', type=str)
  parser.add_argument('save_path', type=str)
  parser.add_argument('-l', '--label_col', type=str, default='label')
  parser.add_argument('-t', '--text_col', type=str, default='text')
  parser.add_argument('-n', '--num_words', type=int, default=10000)

  args = parser.parse_args()
  tokenize(**vars(args))
