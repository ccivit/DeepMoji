from __future__ import print_function, division
import platform
platform.python_version()
import sys
import json
import csv
import numpy as np
import os
import emoji

deep_moji_path = 'gdrive/My Drive/Colab Notebooks/deepmoji'
deep_moji_path = 'deepmoji'
try:
  if second_time_running:
    print(os.listdir(deep_moji_path))
except:
  from google.colab import drive
  drive.mount('/content/gdrive')
  sys.path.insert(0,deep_moji_path)
  sys.path.insert(0,deep_moji_path + '/deepmoji')
  print(os.listdir(deep_moji_path + '/deepmoji'))
  second_time_running = True

from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is a shit']

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]
maxlen = 30
batch_size = 32

print('Loading model from {}.'.format(PRETRAINED_PATH))
print('Model is found? :',os.path.isfile(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

def load_emoji_mapping(emoji_mapping_file):
  emoji_mapping = []
  with open(emoji_mapping_file) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csvreader:
      emoji_mapping.append(', '.join(row))
  return emoji_mapping

def emojify(emoji_token, emoji_mapping):
  emoji = emoji_mapping[int(emoji_token)].split(',')[2]
  return emoji

emoji_mapping_file = os.path.join(deep_moji_path,'emoji_unicode','emoji_unicode.csv')
emoji_map = load_emoji_mapping(emoji_mapping_file)

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
print('Running predictions.')
prob = model.predict(tokenized)


scores = []
for i, t in enumerate(TEST_SENTENCES):
  t_tokens = tokenized[i]
  t_score = [t]
  t_prob = prob[i]
  ind_top = top_elements(t_prob, 5)
  ind_top_emoji = []
  for item in ind_top:
    ind_top_emoji.append(emojify(item,emoji_map))
  t_score.append(sum(t_prob[ind_top]))
  t_score.extend(ind_top_emoji)
  t_score.extend([t_prob[ind] for ind in ind_top])
  scores.append(t_score)

for i in range(len(scores)):
  print(scores[i][0],scores[i][1])
  for j in range(5):
    print('\t',scores[i][j + 2],scores[i][j + 7])
