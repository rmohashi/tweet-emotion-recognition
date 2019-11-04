import os
import re
import click
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from google.cloud import storage

@click.command()
@click.option('--drop_duplicates', '-dd', is_flag=True)
@click.option('--bucket', '-b', type=str, default='twitter-emotion-data')
@click.option('--save_folder', '-s', type=str, default='datasets/twitter-scraper')
def concat_data_from_gcloud(drop_duplicates, bucket, save_folder):
  client = storage.Client()
  bucket = client.get_bucket(bucket)
  save_folder_path = Path(save_folder).resolve()

  print('Downloading files from Cloud Storage:')

  filenames = []
  blobs = list(bucket.list_blobs())
  with tqdm(total=len(blobs)) as t:
    for blob in blobs:
      filenames.append(blob.name)

      destination_path = save_folder_path.joinpath(blob.name)
      blob.download_to_filename(destination_path.as_posix())
      t.update()

  queries = [re.search(r'(#[^.]+|:.+:)', name).group() for name in filenames]
  unique_queries = list(dict.fromkeys(queries))

  print('\nConcating files with the same query')

  with tqdm(total=len(filenames)) as t:
    for query in unique_queries:
      files_to_use = [filename for filename in filenames if re.search(query, filename)]
      data = pd.DataFrame(columns=['id', 'date', 'user', 'text'])
      for filename in files_to_use:
        filepath = save_folder_path.joinpath(filename)
        data = pd.concat([data, pd.read_csv(filepath)])
        os.remove(filepath)
        t.update()
      if drop_duplicates:
        data_without_duplicates = data.drop_duplicates()
      ordered_data = data_without_duplicates.sort_values(by=['id'])
      name = str(max(ordered_data.id)) + '-' + str(min(ordered_data.id)) + '_' + query + '.csv'
      ordered_data.to_csv(save_folder_path.joinpath(name), index=None)

  print('Done!')

if __name__ == '__main__':
  concat_data_from_gcloud()
