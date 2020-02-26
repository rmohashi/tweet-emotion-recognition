import os
import re
import click
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from random import random
from emoji import demojize, emojize

@click.command()
@click.option('--data_dir', '-dd', type=str, default='datasets/twitter-scraper')
def filter_possible_news(data_dir):
  data_path = Path(data_dir).resolve()
  filenames = [x for x in os.listdir(data_path) if x.endswith('.csv')]

  print('Filtering files:')

  with tqdm(total=len(filenames)) as t:
    for filename in filenames:
      filepath = data_path.joinpath(filename)
      df = pd.read_csv(filepath, lineterminator='\n')

      df = df[~df.text.str.contains('http')]
      df.to_csv(filepath, index=None)
      t.update()
  print('Done!')

if __name__ == '__main__':
  filter_possible_news()
