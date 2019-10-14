import click
import pandas as pd
from pathlib import Path
from os import system, name

@click.command()
@click.argument('file', type=str)
@click.option('--save_dir', '-s', type=str, default='datasets/labeled')
@click.option('--start_line', '-s', default=0, type=int)
def label_dataset(file, save_dir, start_line):
  dataset_path = Path(file).resolve()
  data = pd.read_csv(dataset_path)

  save_path = Path(save_dir).resolve()
  output = save_path.joinpath(dataset_path.name)

  emotions = {
    '1': 'joy',
    '2': 'sadness',
    '3': 'anger',
    '4': 'fear'
  }

  start_line = start_line or 0
  for index in range(start_line, len(data)):
    clear()
    print(data.iloc[index].text)
    emotion = input('\n{} - (1)joy (2)sadness (3)anger (4)fear: '.format(index))
    if emotion in emotions:
      data_to_write = data.iloc[index:index+1]
      data_to_write['label'] = emotions[emotion]
      data_to_write.to_csv(output, index=False, header=False, mode='a')
    print("\n")

def clear():
  # for windows
  if name == 'nt':
    _ = system('cls')

  # for mac and linux(here, os.name is 'posix')
  else:
    _ = system('clear')

if __name__ == '__main__':
  label_dataset()
