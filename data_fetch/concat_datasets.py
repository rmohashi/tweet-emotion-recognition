import os
import pandas as pd
from pathlib import Path

def concat_datasets(query, dataset_dir):
  DATASET_DIR = Path(dataset_dir).resolve()
  dir_list = os.listdir(DATASET_DIR)
  if len(dir_list) > 0:
    filenames = pd.Series(dir_list, name='files')
    filenames = filenames[filenames.str.contains(query)]
    file_count = len(filenames)
    print('Found {} files with query "'.format(file_count) + query + '"')
    if file_count > 1:
      df = pd.DataFrame(columns=['id', 'date', 'user', 'text'])
      for filename in filenames:
        filepath = os.path.join(DATASET_DIR, filename)
        df = pd.concat([df, pd.read_csv(filepath)])
        os.remove(filepath)
      name = str(max(df.id)) + '-' + str(min(df.id)) + '_' + query + '.csv'
      df.to_csv(os.path.join(DATASET_DIR, name), index=None)
      print('Done!')

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('query', type=str)
  parser.add_argument('dataset_dir', type=str)

  args = parser.parse_args()
  concat_datasets(**vars(args))