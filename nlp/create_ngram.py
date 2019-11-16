import pickle
import click
import numpy as np
from pathlib import Path

from .dataset import Dataset

def create_ngram_set(input_list, ngram_value=2):
  """
  Extract a set of n-grams from a list of integers.

  >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
  {(4, 9), (4, 1), (1, 4), (9, 4)}

  >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
  [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
  """
  return set(zip(*[input_list[i:] for i in range(ngram_value)]))

@click.command()
@click.argument('dataset_path')
@click.argument('tokenizer_path')
@click.argument('save_dir')
@click.option('--ngram_range', '-n', type=click.IntRange(2, 3, clamp=True))
def create_ngram(dataset_path,
                 tokenizer_path,
                 save_dir,
                 ngram_range=1):
  dataset = Dataset(dataset_path)
  dataset.load()
  dataset.preprocess_texts()
  data = dataset.cleaned_data.copy()

  tokenizer_file = Path(tokenizer_path)
  with tokenizer_file.open('rb') as file:
    tokenizer = pickle.load(file)

  sequences = [text.split() for text in data.text]
  list_tokenized = tokenizer.texts_to_sequences(sequences)

  print('Creating n-grams...')

  # Create set of unique n-gram from the training set.
  ngram_set = set()
  for input_list in list_tokenized:
    for i in range(2, ngram_range + 1):
      set_of_ngram = create_ngram_set(input_list, ngram_value=i)
      ngram_set.update(set_of_ngram)

  # Dictionary mapping n-gram token to a unique integer.
  # Integer values are greater than max_features in order
  # to avoid collision with existing features.
  max_features = len(tokenizer.word_index) + 1
  start_index = max_features + 1
  token_index = {v: k + start_index for k, v in enumerate(ngram_set)}
  index_token = {token_index[k]: k for k in token_index}

  # max_features is the highest integer that could be found in the dataset.
  max_features = np.max(list(index_token.keys())) + 1

  save_name = 'token_indice_' + str(max_features) + '_' + str(ngram_range) + '.pickle'
  save_path = Path(save_dir).resolve().joinpath(save_name)
  with save_path.open('wb') as file:
    pickle.dump(token_index, file)

  print('Saved under: "' + save_path.as_posix() + '"')

if __name__ == '__main__':
  create_ngram()
