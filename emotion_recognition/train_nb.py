import os
import pickle
import pandas as pd
from time import time
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from nlp.dataset import Dataset

def train_nb(filename, label_col='label', text_col='text', validation_split=0.3):
  dataset = Dataset(filename, label_col=label_col, text_col=text_col)
  dataset.load()
  dataset.preprocess_texts()

  data = dataset.cleaned_data.copy()
  train = pd.DataFrame(columns=['label', 'text'])
  validation = pd.DataFrame(columns=['label', 'text'])
  for label in data.label.unique():
    label_data = data[data.label == label]
    train_data, validation_data = train_test_split(label_data, test_size=validation_split)
    train = pd.concat([train, train_data])
    validation = pd.concat([validation, validation_data])

  text_clf = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB())])
  tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
  }

  x_train = train.text
  y_train = train.label
  x_validation = validation.text
  y_validation = validation.label

  print('Running Multinomial Naive Bayes...')
  start = time()
  model = GridSearchCV(text_clf, tuned_parameters, n_jobs=4, cv=10)
  model.fit(x_train, y_train)
  print('Finished in: {} mins'.format(round((time() - start) / 60, 2)))

  print('Testing Model...')
  results = model.predict(x_validation)
  print(classification_report(y_validation, results, digits=4))

  filepath = Path('models/emotion_recognition/nb_model.pickle').resolve()

  with filepath.open('wb') as file:
    pickle.dump(model, file)

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('filename', type=str)
  parser.add_argument('-l', '--label_col', type=str, default='label')
  parser.add_argument('-t', '--text_col', type=str, default='text')
  parser.add_argument('-v', '--validation_split', type=float, default=0.3)

  args = parser.parse_args()
  train_nb(**vars(args))
