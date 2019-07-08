import os
import pickle
import pandas as pd
from pathlib import Path
from nlp.dataset import Dataset
from .models.lstm_conv_model import lstm_conv_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train(dataset_path,
          tokenizer_path,
          label_col='label',
          text_col='text',
          validation_split=0.3,
          embedding_dim=100,
          learning_rate=1e-3,
          epochs=10,
          batch_size=32):
  dataset = Dataset(dataset_path, label_col=label_col, text_col=text_col)
  dataset.load()
  dataset.preprocess_texts()

  tokenizer_file = Path(tokenizer_path).resolve()
  with tokenizer_file.open('rb') as file:
    tokenizer = pickle.load(file)

  data = dataset.cleaned_data.copy()
  train = pd.DataFrame(columns=['label', 'text'])
  validation = pd.DataFrame(columns=['label', 'text'])
  for label in data.label.unique():
    label_data = data[data.label == label]
    train_data, validation_data = train_test_split(label_data, test_size=validation_split)
    train = pd.concat([train, train_data])
    validation = pd.concat([validation, validation_data])

  input_lenght = data.text.apply(lambda text: len(text.split())).max()
  input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
  model = lstm_conv_model(input_lenght, input_dim, embedding_dim=embedding_dim)
  optimizer = Adam(learning_rate)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  train_sequences = [text.split() for text in train.text]
  validation_sequences = [text.split() for text in validation.text]
  list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
  list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)
  x_train = pad_sequences(list_tokenized_train, maxlen=input_lenght)
  x_validation = pad_sequences(list_tokenized_validation, maxlen=input_lenght)

  y_train = pd.get_dummies(train.label)
  y_validation = pd.get_dummies(validation.label)

  model.fit(x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_validation, y_validation))

  model_file = Path(model_dir, 'model_weights.h5')
  model.save_weights(model_file)

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('dataset_path', type=str)
  parser.add_argument('tokenizer_path', type=str)
  parser.add_argument('-l', '--label_col', type=str, default='label')
  parser.add_argument('-t', '--text_col', type=str, default='text')
  parser.add_argument('-v', '--validation_split', type=float, default=0.3)
  parser.add_argument('-ed', '--embedding_dim', type=int, default=100)
  parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
  parser.add_argument('-e', '--epochs', type=int, default=10)
  parser.add_argument('-b', '--batch_size', type=int, default=32)

  args = parser.parse_args()
  train(**vars(args))