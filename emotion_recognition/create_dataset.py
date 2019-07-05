import os
import re
import json
import pandas as pd
from pathlib import Path

def prepare_dataset(files_dir, save_dir):
  FILES_DIR = Path(files_dir).resolve()
  RELATIONS_FILE = Path(os.path.abspath(__file__), '../query_relations.json').resolve()

  with RELATIONS_FILE.open('rb') as file:
    relations = json.load(file)

  def load_csv(filename):
    dataframe = pd.read_csv(os.path.join(FILES_DIR, filename))
    query = re.findall(r'(#[^.csv]+|:.+:)', filename)[0]
    dataframe['label'] = relations[query]
    return dataframe

  data = pd.concat([load_csv(filename) for filename in os.listdir(FILES_DIR)])

  save_path = Path(save_dir, 'dataset.csv').resolve()
  data.to_csv(os.path.join(save_path), index=None)
  print('Saved at: "' + save_path.as_posix() + '"')

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('files_dir', type=str)
  parser.add_argument('save_dir', type=str)

  args = parser.parse_args()
  prepare_dataset(**vars(args))
