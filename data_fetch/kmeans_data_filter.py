import click
import numpy as np
import pandas as pd
from pathlib import Path

from nlp.utils import preprocess

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

@click.command()
@click.argument('file')
@click.option('--output_name', '-o', type=str)
@click.option('--output_dir_path', '-od', default='datasets/kmeans', type=str)
@click.option('--max_features', '-mf', default=8000, type=int)
@click.option('--n_clusters', '-n', default=4, type=int)
@click.option('--n_init', '-ni', default=10, type=int)
@click.option('--max_iter', '-mi', default=300, type=int)
@click.option('--random_state', '-r', default=20, type=int)
def kmeans_data_filter(file, output_name, output_dir_path, max_features,
                       n_clusters, n_init, max_iter, random_state):
  filepath = Path(file).resolve()
  df = pd.read_csv(filepath)

  preprocessed_data = preprocess(df.text)

  tfidf_vectorizer = TfidfVectorizer(
    min_df = 3,
    max_df = 0.95,
    max_features = max_features
  )
  tfidf = tfidf_vectorizer.fit_transform(preprocessed_data)

  print('Running K-means algorithm...')

  clusters = KMeans(
    n_clusters=n_clusters,
    n_init=n_init,
    max_iter=max_iter,
    random_state=random_state
  ).fit_predict(tfidf)

  cluster_to_use = np.bincount(clusters).argmax()

  output_dir = Path(output_dir_path).resolve()
  output_name = output_name or filepath.name
  output_path = output_dir.joinpath(output_name)

  df[clusters == cluster_to_use].to_csv(output_path, index=None)
  print('Finished processing file.')

if __name__ == '__main__':
  kmeans_data_filter()
