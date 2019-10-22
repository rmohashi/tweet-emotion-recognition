import os
import re
import click
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from emoji import demojize

@click.command()
@click.option('--data_dir', '-dd', type=str, default='datasets/twitter-scraper')
def filter_based_on_text_content(data_dir):
  data_path = Path(data_dir).resolve()
  filenames = [x for x in os.listdir(data_path) if x.endswith('.csv')]
  queries = [re.search(r'(#[^.]+|:.+:)', name).group() for name in filenames]
  files = pd.DataFrame(list(zip(filenames, queries)),
                       columns=['filename', 'query'])

  relations_path = Path(os.path.abspath(__file__),
                        '../../query_relations.json').resolve()
  with relations_path.open('rb') as file:
    relations = json.load(file)

  print('Filtering files:')

  with tqdm(total=len(filenames)) as t:
    for _, file in files.iterrows():
      filepath = data_path.joinpath(file.filename)
      df = pd.read_csv(filepath)
      texts = df.text.apply(demojize)
      emotion = relations[file.query]
      to_filter = '|'.join([key for key, val in relations.items() if val != emotion])
      df = df[~texts.str.contains(to_filter, regex=True)]
      df.to_csv(filepath, index=None)
      t.update()
  print('Done!')

if __name__ == '__main__':
  filter_based_on_text_content()
