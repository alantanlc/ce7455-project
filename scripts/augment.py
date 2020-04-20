import argparse
import json
import glob
import os
import shutil

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random

INPUT_DATA_DIR = './data'
OUTPUT_DATA_DIR = './data_wordnet'
TRAIN_FILE_NAME = 'train_*.jsonl'

def augment_sample(sample, n):
  sample = sample.copy()
  new_tokens = word_tokenize(sample['sentence'])

  # Get tokens with pos tag that is one of the following: noun, verb, adjective and adverb
  include_tags = ['NN', 'VB', 'JJ', 'RB']
  replaceable_tokens = [tag[0] for tag in nltk.pos_tag(new_tokens) if tag[1][:2] in include_tags]

  # Exclude option1, option2 and _ tokens 
  exclude_tokens = [sample['option1'], sample['option2'], '_']
  replaceable_tokens = [token for token in replaceable_tokens if token not in exclude_tokens]

  # Randomly replace n tokens with synonyms
  random_token_list = list(set(replaceable_tokens))
  random.shuffle(random_token_list)
  num_replaced = 0
  for random_token in random_token_list:
    synonyms = get_synonyms(random_token)
    if len(synonyms) >= 1:
      synonym = random.choice(list(synonyms))
      new_tokens = [synonym if token == random_token else token for token in new_tokens.copy()]
      num_replaced += 1
    if num_replaced >= n: # only replace up to n words
      break

  sample['sentence'] = ' '.join(new_tokens)

  return sample

def get_synonyms(word):
  synonyms = set()
  for syn in wordnet.synsets(word):
    for l in syn.lemmas():
      synonym = l.name().replace('_', ' ').replace('-', ' ').lower()
      # synonym = ''.join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
      synonyms.add(synonym)
  if word in synonyms:
    synonyms.remove(word)
  return list(synonyms)

def main():
  parser = argparse.ArgumentParser()

  # Parameters
  parser.add_argument('--input_data_dir', default=INPUT_DATA_DIR, type=str, help='The input data dir. Should contain the .jsonl files')
  parser.add_argument('--output_data_dir', default=OUTPUT_DATA_DIR, type=str, help='The output directory where the augmented data sets will be written.')
  parser.add_argument('--train_file_name', default=TRAIN_FILE_NAME, type=str, help='The generic train file name.')
  parser.add_argument('--n_words', default=1, type=int, help='The number of words to be replaced with the synonym.')

  args = parser.parse_args()

  # Recreate data_wordnet directory
  if os.path.exists(args.output_data_dir):
    shutil.rmtree(args.output_data_dir)
  shutil.copytree(args.input_data_dir, args.output_data_dir)

  # Get train file basenames from input data dir
  files = glob.glob(os.path.join(args.input_data_dir, args.train_file_name))
  files = [os.path.basename(file) for file in files]

  # Augment samples in each file 
  for file in files:
    # Load file and parse json
    data = []
    with open(os.path.join(args.input_data_dir, file), 'r') as f:
      for line in f:
        data.append(json.loads(line))

    # Augment each sample
    augmented_data = []
    for d in data:
      d_aug = augment_sample(d, args.n_words)
      augmented_data.append(d_aug)

    # Write data to file in 'data_wordnet' folder
    with open(os.path.join(args.output_data_dir, file), 'w') as f:
      for d in augmented_data:
        f.write(json.dumps(d) + '\n')

if __name__ == '__main__':
  main()
