import os
import re
import click
import json
import pandas as pd
from nlp.utils import preprocess
from tqdm import tqdm
from pathlib import Path
from emoji import demojize

@click.command()
@click.argument('dataset', type=str)
@click.option('--max_length', '-ma', type=int, default=30)
@click.option('--min_length', '-mi', type=int, default=3)
def filter_dataset_using_length(dataset, max_length, min_length):
  dataset_path = Path(dataset).resolve()
  dataset = pd.read_csv(dataset_path)

  print('Filtering dataset:')

  cleaned_text = preprocess(dataset.text)
  dataset = dataset[cleaned_text.apply(lambda x: len(x.split())) <= max_length]
  dataset = dataset[cleaned_text.apply(lambda x: len(x.split())) >= min_length]

  save_path = dataset_path.parent.joinpath('emotion_' + str(len(dataset)) + '.csv')
  dataset.to_csv(save_path, index=None)

  print('File saved under "' + save_path.as_posix() + '"')

if __name__ == '__main__':
  filter_dataset_using_length()
