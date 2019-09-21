import re
import nltk
from time import time
from emoji import demojize

def preprocess(texts, quiet=False, no_emoji=False):
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
  texts = texts.str.replace(pattern, r"\1")

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

  # Filtering emojis if needed
  if no_emoji:
    texts = texts.str.replace(r":\S+:", '')

  if not quiet:
    print("Time to clean up: {:.2f} sec".format(time() - start))

  return texts
