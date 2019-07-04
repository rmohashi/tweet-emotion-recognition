import os
import pickle
import pandas as pd
from pathlib import Path

from .utils import preprocess

def predict_nb(data_file, model_file, save_dir=None, text_col='text', positive=False):
  file_path = Path(data_file).resolve()
  data = pd.read_csv(file_path)
  cleaned_text = preprocess(data[text_col])

  model_path = Path(model_file).resolve()
  with model_path.open('rb') as file:
    model = pickle.load(file)

  result = model.predict(cleaned_text)
  result = result == 4 if positive else result == 0
  result_data = data[result]

  if save_dir:
    save_path = Path(save_dir, file_path.name).resolve()
    result_data.to_csv(save_path, index=None)
    print('File saved under "' + save_path.as_posix() + '"')
  else:
    return result_data

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('data_file', type=str)
  parser.add_argument('model_file', type=str)
  parser.add_argument('save_dir', type=str)
  parser.add_argument('-t', '--text_col', type=str, default='text')
  parser.add_argument('-p', '--positive', action='store_true', default=False)

  args = parser.parse_args()
  predict_nb(**vars(args))
