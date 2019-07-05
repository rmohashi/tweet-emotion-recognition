import re
import nltk
from time import time

def preprocess(texts):
  start = time()
  # Lowercasing
  texts = texts.str.lower()

  # Remove special chars
  texts = texts.str.replace(r"http\S+", "")
  texts = texts.str.replace(r"http", "")
  texts = texts.str.replace(r"@\S+", "")
  texts = texts.str.replace(r"[^a-z\']", " ")

  # Remove repetitions
  pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
  texts = texts.str.replace(pattern, r"\1")

  # Remove multiple whitespaces
  texts = texts.apply(lambda x: ' '.join(x.split()))

  # Transform short negation form
  texts = texts.str.replace(r"(can't|cannot)", 'can not')
  texts = texts.str.replace(r"n't", ' not')

  # Remove stop words
  stopwords = nltk.corpus.stopwords.words('english')
  stopwords.remove('not')
  stopwords.remove('nor')
  stopwords.remove('no')
  texts = texts.apply(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
  )

  print("Time to clean up: {:.2f}".format(time() - start))
  return texts
