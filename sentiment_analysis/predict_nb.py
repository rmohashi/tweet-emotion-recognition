import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from nlp.utils import preprocess

def predict_nb(files_dir, model_file, save_path, text_col='text'):
  FILES_DIR = Path(files_dir).resolve()
  RELATIONS_FILE = Path(os.path.abspath(__file__), '../query_relations.json').resolve()

  with RELATIONS_FILE.open('rb') as file:
    relations = json.load(file)

  model_path = Path(model_file).resolve()
  with model_path.open('rb') as file:
    model = pickle.load(file)

  result_data = []

  for filename in os.listdir(FILES_DIR):
    print('Reading file: "' + filename + '"')
    file_data = pd.read_csv(os.path.join(FILES_DIR, filename))
    cleaned_text = preprocess(file_data[text_col])

    result = model.predict(cleaned_text)
    result = result == 4 if np.bincount(result).argmax() == 4 else result == 0
    file_data = file_data[result]

    query = re.findall(r'(#[^.]+|:.+:)', filename)[0]
    file_data['label'] = relations[query]

    result_data = result_data + [file_data]

  if len(result_data) > 0:
    result_data = pd.concat(result_data)

    path = Path(save_path).resolve()
    result_data.to_csv(path, index=None)

    print('Files saved under "' + save_path + '"')

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('files_dir', type=str)
  parser.add_argument('model_file', type=str)
  parser.add_argument('save_path', type=str)
  parser.add_argument('-t', '--text_col', type=str, default='text')

  args = parser.parse_args()
  predict_nb(**vars(args))
