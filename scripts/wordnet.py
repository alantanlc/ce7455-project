import argparse
import json
import glob
import os

INPUT_DATA_DIR = '../data'
OUTPUT_DATA_DIR = '../data_wordnet'
TRAIN_FILE_NAME = 'train_*.jsonl'

def main():
  parser = argparse.ArgumentParser()

  # Parameters
  parser.add_argument('--input_data_dir', default=INPUT_DATA_DIR, type=str, help='The input data dir. Should contain the .jsonl files')
  parser.add_argument('--output_data_dir', default=OUTPUT_DATA_DIR, type=str, help='The output directory where the augmented data sets will be written.')
  parser.add_argument('--train_file_name', default=TRAIN_FILE_NAME, type=str, help='The generic train file name.')

  args = parser.parse_args()

  # Create output data directory if not exists
  if not os.path.exists(args.output_data_dir):
    os.mkdir(args.output_data_dir)

  # Get train file basenames from input data dir
  files = glob.glob(os.path.join(args.input_data_dir, args.train_file_name))
  files = [os.path.basename(file) for file in files]

  # Augment samples in each file 
  for file in files:
    # Instantiate new data list
    data = []
  
    # Load file and parse json
    with open(os.path.join(args.input_data_dir, file), 'r') as f:
      for line in f:
        data.append(json.loads(line))

    # Write data to file in 'data_wordnet' folder
    with open(os.path.join(args.output_data_dir, file), 'w') as f:
      for d in data:
        f.write(json.dumps(d) + '\n')

if __name__ == '__main__':
  main()
