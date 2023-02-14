---
title: "Effecitve Approaches to Attention-based Neural Machine Translation (2)"
tags: [Pytorch, Deep Learning, Attention]
comments: true
excerpt: Attention
date : 2023-01-06
categories: 
  - paper_review
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : NLP
---

# IV. Implement

## 1. Dataset Load & Library Load


```python
# !wget https://www.statmt.org/europarl/v7/es-en.tgz
# !tar -xf es-en.tgz

# !wget https://download.pytorch.org/tutorial/data.zip
# !unzip data.zip
```


```python
import os
import re
import pickle
import unicodedata
import string
import pandas as pd

import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
from torchtext.data import get_tokenizer

from tqdm import tqdm
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import tensorflow as tf



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
def indexesFromSentence(lang, sentence): #문장을 단어로 분리하고 인덱스를 반환
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence): # 딕셔너리에서 단어에 대한 인덱스를 가져오고 문장 끝에 토큰을 추가
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return indexes


def tensorsFromPair(lang, sen):  # 입력과 출력 문장을 텐서로 변환하여 반환
    ten = tensorFromSentence(lang, sen)


    # input_tensor = tf.keras.preprocessing.sequence.pad_sequences([input_tensor], 
    #                                                              maxlen=MAX_LENGTH,
    #                                                              padding='post', value = -1)
    
    # target_tensor = tf.keras.preprocessing.sequence.pad_sequences([target_tensor], 
    #                                                               maxlen=MAX_LENGTH, 
    #                                                               padding='post', value = -1)
    # ten = torch.tensor(ten, dtype=torch.long, device=device)#.view(-1,1)
    return ten


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded
```


```python
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	pickle.dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load English data
filename = 'europarl-v7.es-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'english.pkl')
# spot check
for i in range(10):
	print(sentences[i])

# load French data
filename = 'europarl-v7.es-en.es'
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'spanish.pkl')
# spot check
for i in range(1):
	print(sentences[i])
```

    Saved: english.pkl
    resumption of the session
    i declare resumed the session of the european parliament adjourned on friday december and i would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period
    although as you will have seen the dreaded millennium bug failed to materialise still the people in a number of countries suffered a series of natural disasters that truly were dreadful
    you have requested a debate on this subject in the course of the next few days during this partsession
    in the meantime i should like to observe a minute s silence as a number of members have requested on behalf of all the victims concerned particularly those of the terrible storms in the various countries of the european union
    please rise then for this minute s silence
    the house rose and observed a minute s silence
    madam president on a point of order
    you will be aware from the press and television that there have been a number of bomb explosions and killings in sri lanka
    one of the people assassinated very recently in sri lanka was mr kumar ponnambalam who had visited the european parliament just a few months ago
    Saved: spanish.pkl
    reanudacion del periodo de sesiones
    


```python
with open('spanish.pkl', 'rb') as f:
    fr_voc = pickle.load(f)

with open('english.pkl', 'rb') as f:
    eng_voc = pickle.load(f)
    
data1 = pd.DataFrame(zip(eng_voc, fr_voc), columns = ['English', 'Spanish'])
data2 = pd.read_csv('data/eng-fra.txt', '\t', names = ['English', 'Spanish'])

data = pd.concat([data1,data2], ignore_index= True, axis = 0)
```

    /usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py:3326: FutureWarning: In a future version of pandas all arguments of read_csv except for the argument 'filepath_or_buffer' will be keyword-only
      exec(code_obj, self.user_global_ns, self.user_ns)
    


```python
data.head()
```





  <div id="df-2fdf8170-8ea1-4502-9fbb-020bc7991c8e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>English</th>
      <th>Spanish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>resumption of the session</td>
      <td>reanudacion del periodo de sesiones</td>
    </tr>
    <tr>
      <th>1</th>
      <td>i declare resumed the session of the european ...</td>
      <td>declaro reanudado el periodo de sesiones del p...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>although as you will have seen the dreaded mil...</td>
      <td>como todos han podido comprobar el gran efecto...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>you have requested a debate on this subject in...</td>
      <td>sus senorias han solicitado un debate sobre el...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>in the meantime i should like to observe a min...</td>
      <td>a la espera de que se produzca de acuerdo con ...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2fdf8170-8ea1-4502-9fbb-020bc7991c8e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2fdf8170-8ea1-4502-9fbb-020bc7991c8e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2fdf8170-8ea1-4502-9fbb-020bc7991c8e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 2.Data Preprocess


```python
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

```


```python
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```


```python
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s%s-%s.txt' % ('data/',lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]


    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

```


```python
MAX_LENGTH = 20

eng_prefixes = [
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ", "I don t", "Do you", "I want", "Are you", "I have", "I think",
       "I can t", "I was", "He is", "I m not", "This is", "I just", "I didn t",
       "I am", "I thought", "I know", "Tom is", "I had", "Did you", "Have you",
       "Can you", "He was", "You don t", "I d like", "It was", "You should",
       "Would you", "I like", "It is", "She is", "You can t", "He has",
       "What do", "If you", "I need", "No one", "You are", "You have",
       "I feel", "I really", "Why don t", "I hope", "I will", "We have",
       "You re not", "You re very", "She was", "I love", "You must", "I can"]
eng_prefixes = (map(lambda x: x.lower(), eng_prefixes))
eng_prefixes = set(eng_prefixes)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(tuple(eng_prefixes))


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
```


```python
def prepareData(lang1, lang2,reverse = False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
```

    Reading lines...
    Read 135842 sentence pairs
    Trimmed to 47571 sentence pairs
    Counting words...
    Counted words:
    fra 10267
    eng 6467
    ['je dispose de ma propre chambre .', 'i have my own room .']
    


```python
input = pd.DataFrame(pairs)[0]
target = pd.DataFrame(pairs)[1]

input_tensor = list(map(lambda x: tensorsFromPair(input_lang,x), input))
target_tensor = list(map(lambda x: tensorsFromPair(output_lang,x), target))


input_tensor = [pad_sequences(x, MAX_LENGTH).tolist() for x in input_tensor]
target_tensor = [pad_sequences(x, MAX_LENGTH).tolist() for x in target_tensor]

input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device)
target_tensor = torch.tensor(target_tensor, dtype=torch.long, device=device)
```

