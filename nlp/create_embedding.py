import click
from pathlib import Path
from gensim.models import Word2Vec

from .dataset import Dataset

@click.command()
@click.argument('filename', type=str)
@click.option('--size', '-s', type=int, default=500)
@click.option('--max_vocab_size', '-m', type=int, default=20000)
@click.option('--workers', '-w', type=int, default=1)
def create_embedding(filename, size, max_vocab_size, workers):
  dataset = Dataset(filename)
  dataset.load()
  dataset.preprocess_texts(stemming=True)

  sentences = dataset.dataframe.cleaned.transform(lambda t: t.split())

  print('Running Word2Vec...')

  embedding = Word2Vec(sentences, size=size, max_vocab_size=max_vocab_size, workers=workers)

  output_dir = Path(filename).resolve().parent
  output_file = output_dir.joinpath('{}_{}d.txt'.format('word2vec', size))
  embedding.wv.save_word2vec_format(output_file)

  print('Word Embedding file saved under: "' + output_file.as_posix() + '"')

if __name__ == '__main__':
  create_embedding()
