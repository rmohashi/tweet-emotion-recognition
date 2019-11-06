import os
import re
import gensim
import pandas as pd

from pathlib import Path
from nlp.dataset import Dataset

def get_topics(query, save_path, method='lda', count=6, num_workers=5):
  preprocessed_docs = get_preprocessed_docs(query, save_path);
  if preprocessed_docs is None:
    return None

  bow_corpus, dictionary = preprocessed_docs
  topics_words = []
  if method == 'lda':
    model = gensim.models.ldamulticore.LdaMulticore(bow_corpus, 
                                                    num_topics=count, 
                                                    id2word=dictionary,                                    
                                                    passes=10,
                                                    workers=num_workers,
                                                    chunksize=100,
                                                    iterations=400)
    topics_words = get_topics_words(model.print_topics())
  elif method == 'lsa' or method == 'lsi':
    model = gensim.models.LsiModel(bow_corpus, 
                                  num_topics=count,
                                  id2word = dictionary)
    topics_words = get_topics_words(model.print_topics(num_topics=count, num_words=8))
  else:
    print('Invalid method "{}"'.format(method))
    return None
  return topics_words

def get_preprocessed_docs(query, save_path):
  FILES_DIR = Path(save_path).resolve()
  dir_list = os.listdir(FILES_DIR)
  filename = [filename for filename in dir_list if query in filename]
  if filename:
    dataset = Dataset('{}/{}'.format(save_path, filename[0]))
    dataset.load()
    dataset.preprocess_texts(lemmatization=True, no_emoji=True)
    tokenized_documents = [tweet.split() for tweet in dataset.dataframe.cleaned]
    dictionary = gensim.corpora.Dictionary(tokenized_documents)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
    return (bow_corpus, dictionary)
  else:
    print('Unable to find dataset for "{}"'.format(query))
    return None

def get_topics_words(topics):
  return [sorted(re.findall(r'"(.*?)"', topic[1])) for topic in topics]
