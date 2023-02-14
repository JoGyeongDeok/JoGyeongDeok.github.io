---
title: "Attention Is All You Need (2)"
tags: [Pytorch, Deep Learning, Attention, Transformer]
comments: true
excerpt: Transformer
date : 2023-02-07
categories: 
  - paper_review
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : NLP
---


# **III. Implement**

## 1. Dataset Load & Library Load



```python
!wget https://www.statmt.org/europarl/v7/es-en.tgz
!tar -xf es-en.tgz

!wget https://download.pytorch.org/tutorial/data.zip
!unzip data.zip
```

    --2023-02-09 13:33:25--  https://www.statmt.org/europarl/v7/es-en.tgz
    Resolving www.statmt.org (www.statmt.org)... 129.215.197.184
    Connecting to www.statmt.org (www.statmt.org)|129.215.197.184|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 195067649 (186M) [application/x-gzip]
    Saving to: ‘es-en.tgz.1’
    
    es-en.tgz.1         100%[===================>] 186.03M  1.16MB/s    in 2m 34s  
    
    2023-02-09 13:35:59 (1.21 MB/s) - ‘es-en.tgz.1’ saved [195067649/195067649]
    
    --2023-02-09 13:36:04--  https://download.pytorch.org/tutorial/data.zip
    Resolving download.pytorch.org (download.pytorch.org)... 99.86.38.96, 99.86.38.72, 99.86.38.37, ...
    Connecting to download.pytorch.org (download.pytorch.org)|99.86.38.96|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2882130 (2.7M) [application/zip]
    Saving to: ‘data.zip.1’
    
    data.zip.1          100%[===================>]   2.75M  --.-KB/s    in 0.05s   
    
    2023-02-09 13:36:05 (60.9 MB/s) - ‘data.zip.1’ saved [2882130/2882130]
    
    Archive:  data.zip
    replace data/eng-fra.txt? [y]es, [n]o, [A]ll, [N]one, [r]ename: A
      inflating: data/eng-fra.txt        
      inflating: data/names/Arabic.txt   
      inflating: data/names/Chinese.txt  
      inflating: data/names/Czech.txt    
      inflating: data/names/Dutch.txt    
      inflating: data/names/English.txt  
      inflating: data/names/French.txt   
      inflating: data/names/German.txt   
      inflating: data/names/Greek.txt    
      inflating: data/names/Irish.txt    
      inflating: data/names/Italian.txt  
      inflating: data/names/Japanese.txt  
      inflating: data/names/Korean.txt   
      inflating: data/names/Polish.txt   
      inflating: data/names/Portuguese.txt  
      inflating: data/names/Russian.txt  
      inflating: data/names/Scottish.txt  
      inflating: data/names/Spanish.txt  
      inflating: data/names/Vietnamese.txt  
    


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
    indexes = [SOS_token]
    indexes.extend(indexesFromSentence(lang, sentence))
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





  <div id="df-987a3a5a-2f7f-45de-8c3b-747654a4dd9e">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-987a3a5a-2f7f-45de-8c3b-747654a4dd9e')"
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
          document.querySelector('#df-987a3a5a-2f7f-45de-8c3b-747654a4dd9e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-987a3a5a-2f7f-45de-8c3b-747654a4dd9e');
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
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS

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
MAX_LENGTH = 100

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
    Trimmed to 47783 sentence pairs
    Counting words...
    Counted words:
    fra 10329
    eng 6508
    ['je dispose de nourriture .', 'i have food .']
    


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

## 3. Transformer

### 3.1 Scaled Dot-Product Attention


```python
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
```

### 3.2  Multi-Head Attention


```python
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

```

### 3.3 Position-wise Feed-Forward Networks


```python
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

```

### 3.4 Positional Encoding


```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
```

### 3.5 Encoder & Decoder


```python
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention


```


```python
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

```

### 3.6 Transforemr


```python
class Transformer(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
```

## 4. Neural Machine Translation with Transformer


```python
def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

```


```python
class CustomDataset(Dataset):
    def __init__(self, input, target = None):
        self.input = input
        self.target = target
        # TODO: convert this into torch code is possible
        self.length = [ torch.sum(x!=0) for x in input]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x_len = self.length[idx]
        if self.target is not None:
            input_tensor, target_tensor = self.input[idx], self.target[idx]
            return input_tensor, target_tensor
        else:
            input_tensor = self.input[idx]
            return input_tensor
```


```python
train_input, valid_input, train_target, valid_target = train_test_split(input_tensor, target_tensor, test_size = 0.2, random_state = 50)

batch_size = 64
train_buffer_size = train_input.shape[0]
train_n_batch = train_buffer_size//batch_size

valid_buffer_size = valid_input.shape[0]
valid_n_batch = valid_buffer_size//batch_size

train_dataset = CustomDataset(train_input, train_target)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True,)
vali_dataset = CustomDataset(valid_input, valid_target)
vali_loader = DataLoader(vali_dataset, batch_size = batch_size, shuffle=False, drop_last=True,)
```


```python
def train(model, optimizer, data_loader, clip = 1):
    model.train()
    total_loss = 0 
    for (batch, (inp, targ)) in enumerate(data_loader):
        loss = 0

        x = inp.to(device)
        y = targ.to(device)
        pred, _ = model(x, y[:,:-1])
        output_dim = pred.shape[-1]
        pred = pred.contiguous().view(-1, output_dim)
        y = y[:,1:].contiguous().view(-1)

        loss = criterion(pred, y)

        total_loss += loss
            
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    total_loss = total_loss/len(data_loader)
    ### TODO: Save checkpoint for model
    print('Train Loss {:.7f}'.format(total_loss))
    return total_loss
```


```python
def validation(model, data_loader):
   
    model.eval()


    total_loss = 0
    with torch.no_grad():
        for (batch, (inp, targ)) in enumerate(data_loader):
            loss = 0
            x = inp.to(device)
            y = targ.to(device)
            pred, _ = model(x, y[:,:-1])
            output_dim = pred.shape[-1]
            pred = pred.contiguous().view(-1, output_dim)
            y = y[:,1:].contiguous().view(-1)

            loss = criterion(pred, y)

            total_loss += loss
    total_loss = total_loss/len(data_loader)

    ### TODO: Save checkpoint for model
    print('Validation Loss {:.4f}'.format(total_loss))
    return total_loss

```


```python
INPUT_DIM = input_lang.n_words
OUTPUT_DIM = output_lang.n_words
HID_DIM = 256
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
SRC_PAD_IDX = 0
TRG_PAD_IDX = 0

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)



model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index = 0)
```


```python
epochs = 20
CLIP = 1
train_loss_list = []
valid_loss_list = []
best_score = 9999
for epoch in range(epochs):
    print('==============={}============'.format(epoch+1))

    start = time.time()
    train_loss = train(model, optimizer, data_loader = train_loader, clip = CLIP)
    valid_loss = validation(model, data_loader = vali_loader)
    train_loss_list.append(train_loss.item())
    valid_loss_list.append(valid_loss.item())
    if best_score > train_loss:
        best_score = train_loss
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Train Best Score :', best_score)

```

    ===============1============
    Train Loss 2.5190506
    Validation Loss 1.7744
    Time taken for 1 epoch 104.8652572631836 sec
    
    Train Best Score : tensor(2.5191, device='cuda:0', grad_fn=<DivBackward0>)
    ===============2============
    Train Loss 1.5245469
    Validation Loss 1.3700
    Time taken for 1 epoch 103.7312285900116 sec
    
    Train Best Score : tensor(1.5245, device='cuda:0', grad_fn=<DivBackward0>)
    ===============3============
    Train Loss 1.1557337
    Validation Loss 1.2037
    Time taken for 1 epoch 103.6811830997467 sec
    
    Train Best Score : tensor(1.1557, device='cuda:0', grad_fn=<DivBackward0>)
    ===============4============
    Train Loss 0.9420360
    Validation Loss 1.1135
    Time taken for 1 epoch 103.74210000038147 sec
    
    Train Best Score : tensor(0.9420, device='cuda:0', grad_fn=<DivBackward0>)
    ===============5============
    Train Loss 0.7950019
    Validation Loss 1.0798
    Time taken for 1 epoch 103.65557408332825 sec
    
    Train Best Score : tensor(0.7950, device='cuda:0', grad_fn=<DivBackward0>)
    ===============6============
    Train Loss 0.6988341
    Validation Loss 1.0557
    Time taken for 1 epoch 103.77540922164917 sec
    
    Train Best Score : tensor(0.6988, device='cuda:0', grad_fn=<DivBackward0>)
    ===============7============
    Train Loss 0.6220177
    Validation Loss 1.0484
    Time taken for 1 epoch 103.8535361289978 sec
    
    Train Best Score : tensor(0.6220, device='cuda:0', grad_fn=<DivBackward0>)
    ===============8============
    Train Loss 0.5734843
    Validation Loss 1.0462
    Time taken for 1 epoch 103.50703167915344 sec
    
    Train Best Score : tensor(0.5735, device='cuda:0', grad_fn=<DivBackward0>)
    ===============9============
    Train Loss 0.5233089
    Validation Loss 1.0290
    Time taken for 1 epoch 103.7166211605072 sec
    
    Train Best Score : tensor(0.5233, device='cuda:0', grad_fn=<DivBackward0>)
    ===============10============
    Train Loss 0.4883089
    Validation Loss 1.0415
    Time taken for 1 epoch 103.75847101211548 sec
    
    Train Best Score : tensor(0.4883, device='cuda:0', grad_fn=<DivBackward0>)
    ===============11============
    Train Loss 0.4648556
    Validation Loss 1.0259
    Time taken for 1 epoch 103.51520133018494 sec
    
    Train Best Score : tensor(0.4649, device='cuda:0', grad_fn=<DivBackward0>)
    ===============12============
    Train Loss 0.4356498
    Validation Loss 1.0403
    Time taken for 1 epoch 103.434326171875 sec
    
    Train Best Score : tensor(0.4356, device='cuda:0', grad_fn=<DivBackward0>)
    ===============13============
    Train Loss 0.4072731
    Validation Loss 1.0400
    Time taken for 1 epoch 103.72106051445007 sec
    
    Train Best Score : tensor(0.4073, device='cuda:0', grad_fn=<DivBackward0>)
    ===============14============
    Train Loss 0.3910976
    Validation Loss 1.0381
    Time taken for 1 epoch 103.64184260368347 sec
    
    Train Best Score : tensor(0.3911, device='cuda:0', grad_fn=<DivBackward0>)
    ===============15============
    Train Loss 0.3766443
    Validation Loss 1.0432
    Time taken for 1 epoch 103.85363006591797 sec
    
    Train Best Score : tensor(0.3766, device='cuda:0', grad_fn=<DivBackward0>)
    ===============16============
    Train Loss 0.3594558
    Validation Loss 1.0495
    Time taken for 1 epoch 104.2496485710144 sec
    
    Train Best Score : tensor(0.3595, device='cuda:0', grad_fn=<DivBackward0>)
    ===============17============
    Train Loss 0.3448482
    Validation Loss 1.0428
    Time taken for 1 epoch 103.56754493713379 sec
    
    Train Best Score : tensor(0.3448, device='cuda:0', grad_fn=<DivBackward0>)
    ===============18============
    Train Loss 0.3290998
    Validation Loss 1.0649
    Time taken for 1 epoch 103.7973120212555 sec
    
    Train Best Score : tensor(0.3291, device='cuda:0', grad_fn=<DivBackward0>)
    ===============19============
    Train Loss 0.3258630
    Validation Loss 1.0589
    Time taken for 1 epoch 103.82871508598328 sec
    
    Train Best Score : tensor(0.3259, device='cuda:0', grad_fn=<DivBackward0>)
    ===============20============
    Train Loss 0.3173423
    Validation Loss 1.0549
    Time taken for 1 epoch 103.70676589012146 sec
    
    Train Best Score : tensor(0.3173, device='cuda:0', grad_fn=<DivBackward0>)
    


```python
x = [t for t in range(1, epochs+1)]

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

# 다중 플롯을 지정 : ax1은 y1에 대한 그래프
fig, ax1 = plt.subplots()
plt.title('Basic NMT')
ax1.plot(x, train_loss_list, color = 'red', alpha = 0.5)
ax1.set_ylabel('train score', color = 'red', rotation = 90)

# ax2는 y2에 대한 그래프, twinx로 x축을 공유
ax2 = ax1.twinx()
ax2.plot(x, valid_loss_list, color = 'blue', alpha = 0.5)
ax2.set_ylabel('valid score', color = 'blue', rotation = 90)

plt.show()

```


    
![png](/images/2023-02-07-Attention_Is_All_You_Need_files/2023-02-07-Attention_Is_All_You_Need_57_0.png)
    


## 4. Evaluation


```python
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 0).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```


```python
def evaluate(model, idx, max_length=MAX_LENGTH):
    src = input_tensor[idx].unsqueeze(0)
    src_mask = model.make_src_mask(src)

    src_output = model.encoder(src, src_mask).to(device)
    trg_idx = [SOS_token]
    decoded_words = []
    with torch.no_grad():
        for i in range(MAX_LENGTH):
            trg_tensor = torch.LongTensor(trg_idx).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            pred, attention = model.decoder(trg_tensor, src_output, trg_mask, src_mask)
            next_word = pred.argmax(2)[:,-1].item()

            trg_idx.append(next_word)
            if next_word  == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[next_word])
        return decoded_words, attention
```


```python
def evaluateRandomly(model, n=10):
    for i in range(n):
        idx = random.choice(range(input.shape[0]))
        print('>', input[idx])
        print('=', target[idx])
        output_words, attentions = evaluate(model, idx, MAX_LENGTH)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
```


```python
evaluateRandomly(model, 10)
```

    > je ne sais pas si je vous l ai jamais dit mais vous avez de beaux yeux .
    = i don t know if i ve ever told you but you have beautiful eyes .
    < i don t know if i ve ever told you have beautiful eyes .
    
    > nous sommes en train de bouillir de l eau .
    = we are boiling water .
    < we are boiling water .
    
    > nous sommes trop faibles .
    = we re too weak .
    < we re too weak .
    
    > tu es une menteuse .
    = you re a liar .
    < you re a liar .
    
    > preferes tu les blondes ou les brunes ?
    = do you prefer blondes or brunettes ?
    < do you prefer blondes or brunettes ?
    
    > je viens d arriver ici la semaine derniere .
    = i just got here last week .
    < i just got here last week .
    
    > je ne veux meme pas savoir .
    = i don t even want to know .
    < i don t even want to know .
    
    > c est une autorite en matiere d humanites .
    = he is an authority on the humanities .
    < he is an authority on the humanities .
    
    > c est tout ce que j emporte avec moi .
    = this is all i m taking with me .
    < this is all i m taking forever with me .
    
    > tu es bon .
    = you are good .
    < you re good .
    
    


```python
def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i][:,:len(translation)+2].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        model, idx)
    print('input =', input[idx])
    print('output =', ' '.join(output_words))
    display_attention(input[idx].split(), output_words, attentions)

```


```python
idx = random.choice(range(input.shape[0]))
evaluateAndShowAttention(idx)
```

    input = je suis desole nous ne pouvons rester plus longtemps .
    output = i m sorry we can t stay any longer .
    


    
![png](/images/2023-02-07-Attention_Is_All_You_Need_files/2023-02-07-Attention_Is_All_You_Need_64_1.png)
    


# Reference

https://cpm0722.github.io/pytorch-implementation/transformer

https://charon.me/posts/pytorch/pytorch_seq2seq_6/#inference
