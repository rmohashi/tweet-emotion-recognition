import re
import nltk
from time import time
from emoji import demojize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def preprocess(texts, quiet=False, stemming=False, lemmatization=False, no_emoji=False):
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

  # Stemming
  if stemming:
    stemmer = SnowballStemmer("english")
    texts = texts.apply(lambda x: stemmer.stem(x))

  # Lemmatization
  if lemmatization:
    texts = texts.apply(lambda x: lemmatize_sentence(x))

  # Filtering emojis if needed
  if no_emoji:
    texts = texts.str.replace(r":\S+:", '')

  if not quiet:
    print("Time to clean up: {:.2f} sec".format(time() - start))

  return texts

def lemmatize_sentence(sentence):
  lemmatizer = WordNetLemmatizer()
  lemmatized_sentence = ([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in sentence.split()])
  return ' '.join(lemmatized_sentence)

def get_wordnet_pos(word):
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {'J': wordnet.ADJ,
              'N': wordnet.NOUN,
              'V': wordnet.VERB,
              'R': wordnet.ADV}
  return tag_dict.get(tag, wordnet.NOUN)