import click
import pandas as pd
from pathlib import Path

@click.command()
@click.argument('file')
@click.argument('output')
@click.argument('emotion')
@click.option('--start_line', '-s', default=0, type=int)
def gold_checker(file, output, emotion, start_line):
  filepath = Path(file).resolve()
  output_path = Path(output).resolve()

  data = pd.read_csv(file)

  start_line = start_line or 0
  for index in range(start_line, len(data)):
    print(data.iloc[index].text)
    to_include = input('Is line {} {}?(y|n):'.format(index, emotion))
    if to_include != 'n':
      data_to_write = data.iloc[index:index+1]
      data_to_write.to_csv(output_path, index=False, header=False, mode='a')
    print("\n")

if __name__ == '__main__':
  gold_checker()