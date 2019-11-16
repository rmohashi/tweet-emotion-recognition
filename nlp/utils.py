import re
import nltk
from time import time
from emoji import demojize
from nltk.stem.snowball import SnowballStemmer

def preprocess(texts, quiet=False, stemming=True, no_emoji=False):
  start = time()
  # Lowercasing
  texts = texts.str.lower()

  # Remove special chars
  texts = texts.str.replace(r"(http|@)\S+", "")
  texts = texts.str.replace(r"&amp", "and")
  texts = texts.apply(demojize)
  texts = texts.str.replace(r"::", ": :")
  texts = texts.str.replace(r"’", "'")
  texts = texts.str.replace(r"[^a-zA-Z\':_]", " ")

  # Remove repetitions
  pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
  texts = texts.str.replace(pattern, r"\1\1")

  # Transform short negation form
  texts = texts.str.replace(r"(can't|cannot)", 'can not')
  texts = texts.str.replace(r"'m", ' am')
  texts = texts.str.replace(r"'s", ' is')
  texts = texts.str.replace(r"n't", ' not')

  # Remove stop words
  stopwords = nltk.corpus.stopwords.words('english')
  stopwords.remove('not')
  stopwords.remove('nor')
  stopwords.remove('no')
  texts = texts.apply(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
  )

  # Stemming
  if stemming:
    stemmer = SnowballStemmer("english")
    texts = texts.apply(lambda x: stemmer.stem(x))

  # Filtering emojis if needed
  if no_emoji:
    texts = texts.str.replace(r":\S+:", '')

  if not quiet:
    print("Time to clean up: {:.2f} sec".format(time() - start))

  return texts

def add_ngram(sequences, token_index, ngram_range=2):
  """
  Augment the input list of list (sequences) by appending n-grams values.

  Example: adding bi-gram
  >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
  >>> token_index = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
  >>> add_ngram(sequences, token_index, ngram_range=2)
  [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

  Example: adding tri-gram
  >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
  >>> token_index = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
  >>> add_ngram(sequences, token_index, ngram_range=3)
  [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
  """
  new_sequences = []
  for input_list in sequences:
    new_list = input_list[:]
    for ngram_value in range(2, ngram_range + 1):
      for i in range(len(new_list) - ngram_value + 1):
        ngram = tuple(new_list[i:i + ngram_value])
        if ngram in token_index:
          new_list.append(token_index[ngram])
    new_sequences.append(new_list)

  return new_sequences
