import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from nlp.dataset import Dataset
from nlp.utils import add_ngram
from .models import NLP_MODEL
from .callbacks import checkpoints, early_stopping, tensorboard, reduce_lr

from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

def train(model_type,
          dataset_path,
          tokenizer_path,
          save_dir,
          glove_embeddings,
          word2vec_embeddings,
          token_index_path,
          ngram_range,
          max_length,
          label_col='label',
          text_col='text',
          validation_split=0.3,
          embedding_dim=100,
          input_length=100,
          learning_rate=1e-3,
          epochs=10,
          batch_size=32):
  dataset = Dataset(dataset_path, label_col=label_col, text_col=text_col)
  dataset.load()
  dataset.preprocess_texts(stemming=True)

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

  embedding_layer = None
  if(glove_embeddings):
    embeddings_index = {}
    f = Path(os.getenv("GLOVE_EMBEDDINGS")).open()
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()

    embedding_dim = int(os.getenv("EMBEDDING_DIM"))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=True)


  if(word2vec_embeddings):
    embedding = KeyedVectors.load_word2vec_format(os.getenv("WORD2VEC_EMBEDDINGS"))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding.vector_size))
    for word, i in tokenizer.word_index.items():
      try:
        embedding_vector = embedding.get_vector(word)
        embedding_matrix[i] = embedding_vector
      except:
        pass

    embedding_layer = Embedding(
      input_dim=len(tokenizer.word_index) + 1,
      output_dim=embedding.vector_size,
      weights=[embedding_matrix],
      input_length=input_length,
      trainable=False,
      input_shape=(input_length,)
    )

  input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)

  train_sequences = [text.split() for text in train.text]
  validation_sequences = [text.split() for text in validation.text]
  list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
  list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)

  if (model_type == 'fasttext'):
    with Path(token_index_path).resolve().open('rb') as file:
      token_index = pickle.load(file)
    list_tokenized_train = add_ngram(train_sequences, token_index, ngram_range)
    list_tokenized_validation = add_ngram(validation_sequences, token_index, ngram_range)
    input_dim = max_length

  model = NLP_MODEL[model_type](input_length, input_dim, embedding_layer, embedding_dim=embedding_dim)
  optimizer = Adam(learning_rate)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  print(model.summary())

  x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
  x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)

  y_train = train.label.replace(4, 1)
  y_validation = validation.label.replace(4, 1)

  model_name = model_type + '_' + str(embedding_dim) + '_' + str(input_length)
  checkpoint_path = os.path.join(save_dir, 'checkpoints', model_name + '_{epoch:02d}-{val_acc:.4f}.h5')
  log_dir = os.path.join(save_dir, 'logs', model_name)

  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation),
    callbacks=[checkpoints(checkpoint_path), tensorboard(log_dir, batch_size), early_stopping(10), reduce_lr(5)]
  )

  model_file = Path(save_dir, model_name + '.h5').resolve()
  model.save_weights(model_file.as_posix())

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('model_type', type=str, choices=['lstm', 'lstm_conv', 'cnn', 'fasttext'])
  parser.add_argument('dataset_path', type=str)
  parser.add_argument('tokenizer_path', type=str)
  parser.add_argument('save_dir', type=str)
  parser.add_argument('-l', '--label_col', type=str, default='label')
  parser.add_argument('-t', '--text_col', type=str, default='text')
  parser.add_argument('-v', '--validation_split', type=float, default=0.3)
  parser.add_argument('-ge', '--glove_embeddings', action='store_true', default=False)
  parser.add_argument('-we', '--word2vec_embeddings', action='store_true', default=False)
  parser.add_argument('-ti', '--token_index_path', type=str)
  parser.add_argument('-nr', '--ngram_range', type=int)
  parser.add_argument('-ml', '--max_length', type=int)
  parser.add_argument('-ed', '--embedding_dim', type=int, default=100)
  parser.add_argument('-i', '--input_length', type=int, default=100)
  parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
  parser.add_argument('-e', '--epochs', type=int, default=10)
  parser.add_argument('-b', '--batch_size', type=int, default=32)

  args = parser.parse_args()
  train(**vars(args))
