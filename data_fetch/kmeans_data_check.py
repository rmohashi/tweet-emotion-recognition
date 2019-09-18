import click
import numpy as np
import pandas as pd
from pathlib import Path

from nlp.utils import preprocess

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

@click.command()
@click.argument('file')
@click.argument('output')
@click.option('--max_features', '-mf', default=8000, type=int)
@click.option('--n_clusters', '-n', default=4, type=int)
@click.option('--init_size', '-i', default=1024, type=int)
@click.option('--batch_size', '-b', default=2048, type=int)
@click.option('--random_state', '-r', default=20, type=int)
def kmeans_data_check(file, output, max_features, n_clusters,
                      init_size, batch_size, random_state):
  filepath = Path(file).resolve()
  output_path = Path(output).resolve()
  df = pd.read_csv(filepath)

  preprocessed_data = preprocess(df.text)

  tfidf_vectorizer = TfidfVectorizer(
    min_df = 3,
    max_df = 0.95,
    max_features = 8000
  )
  tfidf = tfidf_vectorizer.fit_transform(preprocessed_data)

  clusters = MiniBatchKMeans(
    n_clusters=8,
    init_size=1024,
    batch_size=2048,
    random_state=20
  ).fit_predict(tfidf)

  cluster_to_use = np.bincount(clusters).argmax()
  print(cluster_to_use)
  df[clusters == cluster_to_use].to_csv(output_path, index=None)
  print('Finished processing file.')

if __name__ == '__main__':
  kmeans_data_check()
