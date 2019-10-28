import os
import re
import json
import click
import pandas as pd
from tqdm import tqdm
from pathlib import Path

@click.command()
@click.option('--files_dir', '-f', type=str, default='datasets/twitter-scraper')
@click.option('--save_dir', '-s', type=str, default='datasets/emotion_recognition')
def build_emotion_dataset(files_dir, save_dir):
  files_path = Path(files_dir).resolve()
  relations_path = Path(os.path.abspath(__file__), '../../query_relations.json').resolve()

  with relations_path.open('rb') as file:
    relations = json.load(file)

  result_data = []
  print('Reading files:')

  filenames = os.listdir(files_path)
  with tqdm(total=len(filenames)) as t:
    for filename in filenames:

      query = re.findall(r'(#[^.]+|:.+:)', filename)[0]
      emotion = relations[query]

      file_data = pd.read_csv(files_path.joinpath(filename))
      file_data.insert(0, 'label', emotion)
      result_data = result_data + [file_data]
      t.update()

  if len(result_data) > 0:
    result_data = pd.concat(result_data)

    filename = 'emotions_' + str(len(result_data)) + '.csv'
    save_path = Path(save_dir).resolve().joinpath(filename)

    result_data.to_csv(save_path, index=None)
    print('Files saved under "' + save_path.as_posix() + '"')

if __name__ == '__main__':
  build_emotion_dataset()
