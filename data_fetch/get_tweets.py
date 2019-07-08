import os
import re

import pandas as pd
from pathlib import Path
from emoji import emojize
from tweepy import TweepError

from .connection import Connection

def get_tweets(query, save_dir=None, max_requests=10, count=100, newer=False):
  connection = Connection()
  connection.load()

  max_id, min_id = get_bounding_ids(query, save_dir) if save_dir else (None, -1)
  q = emojize(query) + ' -filter:retweets'
  searched_tweets = []
  last_id = -1 if newer else min_id
  since_id = max_id if newer else None
  request_count = 0
  while request_count < max_requests:
    try:
      new_tweets = connection.api.search(q=q,
                                         lang='en',
                                         count=count,
                                         max_id=str(last_id - 1),
                                         since_id=str(since_id),
                                         tweet_mode='extended')
      if not new_tweets:
          break
      searched_tweets.extend(new_tweets)
      last_id = new_tweets[-1].id
      request_count += 1
    except TweepError as e:
      print(e)
      break

  data = []
  for tweet in searched_tweets:
    data.append([tweet.id, tweet.created_at, tweet.user.screen_name, tweet.full_text])

  df = pd.DataFrame(data=data, columns=['id', 'date', 'user', 'text'])
  print(str(len(data)) + ' ' + query + ' tweets')

  if save_dir and data:
    PATH = Path(save_dir).resolve()
    filename = str(data[0][0]) + '-' + str(last_id) + '_' + query + '.csv'
    df.to_csv(os.path.join(PATH, filename), index=None)
    print('Saved under: "' + PATH.as_posix() + '"')
  else:
    return df

def get_bounding_ids(query, save_dir):
  SAVE_DIR = Path(save_dir).resolve()
  dir_list = os.listdir(SAVE_DIR)
  if len(dir_list) > 0:
    filenames = pd.Series(dir_list, name='files')
    filenames = filenames[filenames.str.contains(query)]
    ids = []
    for filename in filenames:
      file_ids = re.findall(r'\d+', filename)
      ids += file_ids[0:2]
    if len(ids) > 0:
      return (int(max(ids)), int(min(ids)))
  return (None, -1)

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('query', type=str)
  parser.add_argument('-s', '--save_dir', type=str)
  parser.add_argument('-m', '--max_requests', type=int, default=10)
  parser.add_argument('-c', '--count', type=int, default=100)
  parser.add_argument('-n', '--newer', action='store_true', default=False)

  args = parser.parse_args()
  get_tweets(**vars(args))
