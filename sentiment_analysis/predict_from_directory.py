import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nlp.utils import preprocess
from .models import NLP_MODEL

def predict_from_directory(files_dir,
                           model_weights_file,
                           model_type,
                           tokenizer_file,
                           save_path,
                           embedding_dim=100,
                           text_col='text'):
  FILES_DIR = Path(files_dir).resolve()
  RELATIONS_FILE = Path(os.path.abspath(__file__), '../query_relations.json').resolve()

  with RELATIONS_FILE.open('rb') as file:
    relations = json.load(file)

  tokenizer_path = Path(tokenizer_file).resolve()
  with tokenizer_path.open('rb') as file:
    tokenizer = pickle.load(file)

  weights_path = Path(model_weights_file).resolve()
  input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
  model = NLP_MODEL[model_type](100, input_dim, embedding_dim=embedding_dim)
  model.load_weights(weights_path.as_posix())

  emotion_data_dict = {}

  for filename in os.listdir(FILES_DIR):
    print('Reading file: "' + filename + '"')

    query = re.findall(r'(#[^.]+|:.+:)', filename)[0]
    emotion = relations[query]

    file_data = pd.read_csv(os.path.join(FILES_DIR, filename))
    dict_data = emotion_data_dict[emotion] if emotion in emotion_data_dict else None
    emotion_data_dict[emotion] = pd.concat([dict_data, file_data])

  result_data = []

  for emotion, dataset in emotion_data_dict.items():
    print('Processing "' + emotion + '" data...')

    cleaned_texts = preprocess(dataset[text_col], quiet=True)
    predict_sequences = [text.split() for text in cleaned_texts]
    list_tokenized_predict = tokenizer.texts_to_sequences(predict_sequences)
    x_predict = pad_sequences(list_tokenized_predict, maxlen=100)

    result = model.predict(x_predict)
    mean = np.mean(result)
    std = np.std(result)
    low, high = get_score_range(mean)
    print("\tScore Range: {:4f} - {:4f}".format(low, high))
    dataset = dataset[np.all([(result >= low), (result <= high)], axis=0)]
    dataset.insert(0, 'label', emotion)

    result_data = result_data + [dataset]

  if len(result_data) > 0:
    result_data = pd.concat(result_data)

    path = Path(save_path).resolve()
    result_data.to_csv(path, index=None)

    print('Files saved under "' + save_path + '"')

def get_score_range(mean):
  if mean < 0.5:
    return (0.0, mean + 0.05)
  return (mean - 0.05, 1.0)

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('files_dir', type=str)
  parser.add_argument('model_weights_file', type=str)
  parser.add_argument('model_type', type=str, choices=['gru', 'lstm_conv'])
  parser.add_argument('tokenizer_file', type=str)
  parser.add_argument('save_path', type=str)
  parser.add_argument('-e', '--embedding_dim', type=int, default=100)
  parser.add_argument('-t', '--text_col', type=str, default='text')

  args = parser.parse_args()
  predict_from_directory(**vars(args))
