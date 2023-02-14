---
title: "Otto Group Product Classification Challenge"
tags: [Data Anaylsis, Code Review, Machine Learning]
comments: true
date : 2021-11-15
categories: 
  - blog
excerpt: Kaggle 코드 리뷰
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

Kaggle Otto Group Product Classification Challenge 대회의 **HOXOSH**의 코드를 참조하였다. **PCA**를 통해 10개의 주성분으로 데이터를 변환하였고, Tensorflow 모듈을 사용하여 간단한 **신경망** 모델을 사용하였다. 5-kfold로 10epochs씩 학습하였다.


<b><a href = 'https://www.kaggle.com/cashfeg/hoxosh-problem-with-pca'>코드 참조</a></b>


# 1.Library & Data Load


```python
from google.colab import drive
drive.mount('/content/drive')     
```

    Mounted at /content/drive
    


```python
!cp /content/drive/MyDrive/Kaggle/kaggle.json /root/.kaggle/
#!kaggle competitions list
!kaggle competitions download -c otto-group-product-classification-challenge
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.4)
    Downloading train.csv.zip to /content
      0% 0.00/1.69M [00:00<?, ?B/s]
    100% 1.69M/1.69M [00:00<00:00, 27.8MB/s]
    Downloading test.csv.zip to /content
      0% 0.00/4.00M [00:00<?, ?B/s]
    100% 4.00M/4.00M [00:00<00:00, 65.8MB/s]
    Downloading sampleSubmission.csv.zip to /content
      0% 0.00/369k [00:00<?, ?B/s]
    100% 369k/369k [00:00<00:00, 5.94MB/s]
    

<img src = 'https://drive.google.com/uc?id=1_HCs5NiqGpPzMiaVZVHkOv1b0WV7r0Em' height = 1000 width = 700>

**주성분 분석**
 - 데이터에서 주성분 벡터를 찾아 데이터의 차원을 축소시키는 기법
3개 이상의 다차원 데이터에서 Principal Component를 추출하여 그 특징을 추출하거나 패턴을 찾는 통계 기반 알고리즘
 - 데이터의 변동성이 최대가 되는 축을 찾아 주성분으로 결정
 - 분산이 가장 큰 방향이 데이터에서 가장 많은 정도를 담고 있는방향
어떤 벡터로 투영된 데이터의 분산 클수록 데이터를 의미 있게 분석
투영했을 때 데이터 포인트들의 분산이 큰 벡터를 찾는 것이 목적

<img src = 'https://drive.google.com/uc?id=16YYDm-cSraUZzozd6IZKM-ptxamvppem' height = 1000 width = 700>

변수간의 스케일이 차이가 나면 스케일 큰 변수가 주성분에 영향을 많이 주기 때문에 주성분 분석 전에 변수를 표준화나 정규화시켜주는 것이 좋다.

Python에서는 사이킷런 패키지의 StandardScaler모듈을 통해 변수들을 정규화시킬 수 있다.


```python
!unzip "train.csv.zip" -d ""
!unzip "test.csv.zip" -d ""
!unzip "sampleSubmission.csv.zip" -d ""
```

    Archive:  train.csv.zip
      inflating: train.csv               
    Archive:  test.csv.zip
      inflating: test.csv                
    Archive:  sampleSubmission.csv.zip
      inflating: sampleSubmission.csv    
    


```python
!pip install umap
```

    Collecting umap
      Downloading umap-0.1.1.tar.gz (3.2 kB)
    Building wheels for collected packages: umap
      Building wheel for umap (setup.py) ... [?25l[?25hdone
      Created wheel for umap: filename=umap-0.1.1-py3-none-any.whl size=3564 sha256=583084dbe145598054ede11b2506142d22957adf9121c4f454e1fdaa594e5051
      Stored in directory: /root/.cache/pip/wheels/65/55/85/945cfb3d67373767e4dc3e9629300a926edde52633df4f0efe
    Successfully built umap
    Installing collected packages: umap
    Successfully installed umap-0.1.1
    


```python
from typing import Any, Dict
import umap
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf
from sklearn.decomposition import PCA

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape, LayerNormalization, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dropout

from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
import xgboost as xgb
```


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sampleSubmission.csv')
```

# 2. Data 전처리


```python
def docking(train: pd.DataFrame, test: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    print("docking")
    dict_a: Dict
    dict_a = {"Class_1": 0,
              "Class_2": 1,
              "Class_3": 2,
              "Class_4": 3,
              "Class_5": 4,
              "Class_6": 5,
              "Class_7": 6,
              "Class_8": 7,
              "Class_9": 8,
              }
    target = train["target"].map(dict_a)
    con_df = pd.concat([train, test], sort=False)
    """
    targetは Class_n　→ n-1　にした（1-9だったので）
    trainのcolumnsは feat_n → n にする
    """
    con_df = con_df.drop(["id", "target"], axis=1)
    con_df.columns = con_df.columns.map(lambda x: int(x[5:]))
    con_df = np.log1p(con_df)

    return con_df, target

def split_data(df, df_pca, df_features, target, join_pca: bool):
    # df = np.log1p(df)
    # df = pd.concat([df, df_pca], axis=1, join="inner")
    if join_pca:
        # df = pd.concat([df, df_pca], axis=1, join_axes=[df.index])
        df = pd.concat([df.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
    # df = df_pca.copy()

    # df = pd.concat([df, df_features], axis=1, join_axes=[df.index])
    df = pd.concat([df.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
    train_df = df[:len(target)]
    test_df = df[len(target):]
    return train_df, test_df

def make_features(df: pd.DataFrame):
    memo = pd.DataFrame()
    memo["count_zero"] = df[df == 0].count(axis=1)
    # memo["count_one"] = df[df == 1].count(axis=1)
    return memo
```


```python
def do_pca(df: pd.DataFrame, target: pd.Series):
    n = 10
    pca = PCA(n_components=n)
    pca.fit(df[:len(target)])
    # df_pca = pca.fit_transform(df)
    df_pca = pca.transform(df)
    n_name = [f"pca{i}" for i in range(n)]
    df_pca = pd.DataFrame(df_pca, columns=n_name)
    return df_pca
```


```python
df, target = docking(train, test)
df_pca = do_pca(df, target)
df_features = make_features(df)
df_train, df_test = split_data(df, df_pca, df_features, target, join_pca=True)
```

    docking
    


```python
df_train
```





  <div id="df-355c5119-c3e2-4538-bd96-baf381dde928">
    <div class="">
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>...</th>
      <th>65</th>
      <th>66</th>
      <th>67</th>
      <th>68</th>
      <th>69</th>
      <th>70</th>
      <th>71</th>
      <th>72</th>
      <th>73</th>
      <th>74</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>count_zero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>1.609438</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>...</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>2.079442</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.421158</td>
      <td>0.635386</td>
      <td>-0.584677</td>
      <td>0.037974</td>
      <td>-1.784932</td>
      <td>-1.376718</td>
      <td>1.172158</td>
      <td>-0.955312</td>
      <td>1.124444</td>
      <td>-0.502617</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-0.948499</td>
      <td>-0.205642</td>
      <td>-1.632092</td>
      <td>0.527130</td>
      <td>-0.083793</td>
      <td>0.442409</td>
      <td>-0.065884</td>
      <td>0.549451</td>
      <td>-0.481483</td>
      <td>-0.538074</td>
      <td>83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.945910</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-1.208384</td>
      <td>0.059220</td>
      <td>-1.073091</td>
      <td>0.146188</td>
      <td>-0.752392</td>
      <td>-0.088797</td>
      <td>-0.419565</td>
      <td>0.782941</td>
      <td>0.347065</td>
      <td>-0.298901</td>
      <td>85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>1.945910</td>
      <td>0.693147</td>
      <td>1.791759</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.079442</td>
      <td>1.098612</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.077537</td>
      <td>0.0</td>
      <td>2.397895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>1.791759</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.609438</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>3.135494</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.940020</td>
      <td>1.640381</td>
      <td>-0.169423</td>
      <td>-0.407307</td>
      <td>0.515537</td>
      <td>-0.241580</td>
      <td>0.295070</td>
      <td>1.298766</td>
      <td>0.839002</td>
      <td>0.053846</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-0.958956</td>
      <td>-0.297528</td>
      <td>-1.278163</td>
      <td>0.918902</td>
      <td>-0.684443</td>
      <td>0.581809</td>
      <td>0.063009</td>
      <td>1.067866</td>
      <td>-0.255079</td>
      <td>0.019961</td>
      <td>82</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>61873</th>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.302585</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.079442</td>
      <td>0.0</td>
      <td>1.386294</td>
      <td>1.945910</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>4.189655</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>1.609438</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>1.098612</td>
      <td>...</td>
      <td>1.609438</td>
      <td>2.484907</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.401197</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>2.711217</td>
      <td>1.651381</td>
      <td>3.892783</td>
      <td>-0.701968</td>
      <td>-2.460966</td>
      <td>-0.962636</td>
      <td>-0.273677</td>
      <td>0.561411</td>
      <td>-1.822165</td>
      <td>-0.653342</td>
      <td>51</td>
    </tr>
    <tr>
      <th>61874</th>
      <td>1.609438</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>1.791759</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>2.484907</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>-1.105991</td>
      <td>0.215849</td>
      <td>0.448056</td>
      <td>0.194435</td>
      <td>-0.862646</td>
      <td>-1.162472</td>
      <td>-0.689592</td>
      <td>1.644892</td>
      <td>-0.465882</td>
      <td>-0.710806</td>
      <td>74</td>
    </tr>
    <tr>
      <th>61875</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.944439</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-1.065043</td>
      <td>1.542719</td>
      <td>0.133696</td>
      <td>-0.637054</td>
      <td>-1.109646</td>
      <td>-0.833269</td>
      <td>-1.553857</td>
      <td>0.941141</td>
      <td>0.912494</td>
      <td>-0.517700</td>
      <td>78</td>
    </tr>
    <tr>
      <th>61876</th>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.791759</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.945910</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>2.397895</td>
      <td>0.0</td>
      <td>0.079951</td>
      <td>0.757954</td>
      <td>0.006619</td>
      <td>-0.460323</td>
      <td>-1.824516</td>
      <td>-0.883310</td>
      <td>-0.755696</td>
      <td>0.293008</td>
      <td>-0.442748</td>
      <td>-0.700269</td>
      <td>71</td>
    </tr>
    <tr>
      <th>61877</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.386294</td>
      <td>0.0</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.302585</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.397895</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.0</td>
      <td>0.920260</td>
      <td>1.070147</td>
      <td>0.269113</td>
      <td>-0.048444</td>
      <td>-2.539279</td>
      <td>-0.966442</td>
      <td>-0.315356</td>
      <td>-0.039758</td>
      <td>-0.023687</td>
      <td>-1.274666</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
<p>61878 rows × 104 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-355c5119-c3e2-4538-bd96-baf381dde928')"
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
          document.querySelector('#df-355c5119-c3e2-4538-bd96-baf381dde928 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-355c5119-c3e2-4538-bd96-baf381dde928');
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




# 3. 모델링

```python
def nn_train_model(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame
):
    n_splits = 5
    num_class = 9
    epochs = 10
    lr_init = 0.01
    bs = 256
    num_features = df.shape[1]
    folds = KFold(n_splits=n_splits, random_state=71, shuffle=True)

    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),
        Dense(128, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(128, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(num_class, activation="softmax")
    ])
    
    # print(model.summary())
    optimizer = tf.keras.optimizers.Adam(lr=lr_init, decay=0.0001)

    """callbacks"""
    callbacks = []
    # callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    preds = np.zeros((test.shape[0], num_class))
    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        # train_x = np.reshape(train_x, (-1, num_features, 1))
        # val_x = np.reshape(val_x, (-1, num_features, 1))
        model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2, batch_size=bs,
                  callbacks=callbacks)
        preds += model.predict(test.values) / n_splits

    return preds
```


```python
pred1 = nn_train_model(df_train, target, df_test)
```

    /usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(Adam, self).__init__(name, **kwargs)
    

    Epoch 1/10
    194/194 - 3s - loss: 0.7828 - accuracy: 0.7143 - val_loss: 0.6839 - val_accuracy: 0.7442 - 3s/epoch - 13ms/step
    Epoch 2/10
    194/194 - 1s - loss: 0.6754 - accuracy: 0.7459 - val_loss: 0.7619 - val_accuracy: 0.6917 - 1s/epoch - 5ms/step
    Epoch 3/10
    194/194 - 1s - loss: 0.6526 - accuracy: 0.7502 - val_loss: 0.8036 - val_accuracy: 0.6705 - 1s/epoch - 5ms/step
    Epoch 4/10
    194/194 - 1s - loss: 0.6422 - accuracy: 0.7551 - val_loss: 0.7015 - val_accuracy: 0.7258 - 1s/epoch - 6ms/step
    Epoch 5/10
    194/194 - 1s - loss: 0.6352 - accuracy: 0.7566 - val_loss: 0.8287 - val_accuracy: 0.7033 - 1s/epoch - 6ms/step
    Epoch 6/10
    194/194 - 1s - loss: 0.6256 - accuracy: 0.7580 - val_loss: 0.6349 - val_accuracy: 0.7455 - 1s/epoch - 6ms/step
    Epoch 7/10
    194/194 - 1s - loss: 0.6236 - accuracy: 0.7613 - val_loss: 0.6870 - val_accuracy: 0.7383 - 1s/epoch - 6ms/step
    Epoch 8/10
    194/194 - 1s - loss: 0.6201 - accuracy: 0.7623 - val_loss: 0.6331 - val_accuracy: 0.7588 - 1s/epoch - 6ms/step
    Epoch 9/10
    194/194 - 1s - loss: 0.6150 - accuracy: 0.7637 - val_loss: 0.6116 - val_accuracy: 0.7625 - 1s/epoch - 5ms/step
    Epoch 10/10
    194/194 - 1s - loss: 0.6122 - accuracy: 0.7649 - val_loss: 0.5870 - val_accuracy: 0.7717 - 1s/epoch - 6ms/step
    Epoch 1/10
    194/194 - 1s - loss: 0.6076 - accuracy: 0.7660 - val_loss: 0.5852 - val_accuracy: 0.7777 - 1s/epoch - 6ms/step
    Epoch 2/10
    194/194 - 1s - loss: 0.6054 - accuracy: 0.7667 - val_loss: 0.5745 - val_accuracy: 0.7813 - 1s/epoch - 6ms/step
    Epoch 3/10
    194/194 - 1s - loss: 0.6020 - accuracy: 0.7694 - val_loss: 0.6411 - val_accuracy: 0.7419 - 1s/epoch - 6ms/step
    Epoch 4/10
    194/194 - 1s - loss: 0.5973 - accuracy: 0.7708 - val_loss: 0.6445 - val_accuracy: 0.7547 - 1s/epoch - 6ms/step
    Epoch 5/10
    194/194 - 1s - loss: 0.5964 - accuracy: 0.7704 - val_loss: 0.5949 - val_accuracy: 0.7716 - 1s/epoch - 6ms/step
    Epoch 6/10
    194/194 - 1s - loss: 0.5959 - accuracy: 0.7689 - val_loss: 0.5902 - val_accuracy: 0.7766 - 1s/epoch - 7ms/step
    Epoch 7/10
    194/194 - 1s - loss: 0.5916 - accuracy: 0.7714 - val_loss: 0.6039 - val_accuracy: 0.7692 - 1s/epoch - 6ms/step
    Epoch 8/10
    194/194 - 1s - loss: 0.5896 - accuracy: 0.7738 - val_loss: 0.5863 - val_accuracy: 0.7763 - 1s/epoch - 6ms/step
    Epoch 9/10
    194/194 - 1s - loss: 0.5897 - accuracy: 0.7718 - val_loss: 0.5634 - val_accuracy: 0.7825 - 1s/epoch - 6ms/step
    Epoch 10/10
    194/194 - 1s - loss: 0.5862 - accuracy: 0.7748 - val_loss: 0.5819 - val_accuracy: 0.7771 - 1s/epoch - 6ms/step
    Epoch 1/10
    194/194 - 1s - loss: 0.5887 - accuracy: 0.7722 - val_loss: 0.5966 - val_accuracy: 0.7708 - 1s/epoch - 6ms/step
    Epoch 2/10
    194/194 - 1s - loss: 0.5859 - accuracy: 0.7757 - val_loss: 0.5881 - val_accuracy: 0.7773 - 1s/epoch - 6ms/step
    Epoch 3/10
    194/194 - 1s - loss: 0.5847 - accuracy: 0.7743 - val_loss: 0.5649 - val_accuracy: 0.7827 - 1s/epoch - 6ms/step
    Epoch 4/10
    194/194 - 1s - loss: 0.5804 - accuracy: 0.7735 - val_loss: 0.5529 - val_accuracy: 0.7857 - 1s/epoch - 5ms/step
    Epoch 5/10
    194/194 - 1s - loss: 0.5798 - accuracy: 0.7741 - val_loss: 0.5505 - val_accuracy: 0.7864 - 1s/epoch - 5ms/step
    Epoch 6/10
    194/194 - 1s - loss: 0.5774 - accuracy: 0.7765 - val_loss: 0.5488 - val_accuracy: 0.7797 - 1s/epoch - 6ms/step
    Epoch 7/10
    194/194 - 1s - loss: 0.5748 - accuracy: 0.7776 - val_loss: 0.5516 - val_accuracy: 0.7850 - 1s/epoch - 6ms/step
    Epoch 8/10
    194/194 - 1s - loss: 0.5767 - accuracy: 0.7775 - val_loss: 0.5658 - val_accuracy: 0.7861 - 1s/epoch - 6ms/step
    Epoch 9/10
    194/194 - 1s - loss: 0.5722 - accuracy: 0.7782 - val_loss: 0.5670 - val_accuracy: 0.7851 - 1s/epoch - 5ms/step
    Epoch 10/10
    194/194 - 1s - loss: 0.5738 - accuracy: 0.7760 - val_loss: 0.5375 - val_accuracy: 0.7923 - 1s/epoch - 6ms/step
    Epoch 1/10
    194/194 - 1s - loss: 0.5744 - accuracy: 0.7775 - val_loss: 0.5732 - val_accuracy: 0.7803 - 1s/epoch - 6ms/step
    Epoch 2/10
    194/194 - 1s - loss: 0.5724 - accuracy: 0.7808 - val_loss: 0.5221 - val_accuracy: 0.7922 - 1s/epoch - 6ms/step
    Epoch 3/10
    194/194 - 1s - loss: 0.5715 - accuracy: 0.7799 - val_loss: 0.5548 - val_accuracy: 0.7852 - 1s/epoch - 6ms/step
    Epoch 4/10
    194/194 - 1s - loss: 0.5699 - accuracy: 0.7801 - val_loss: 0.5606 - val_accuracy: 0.7758 - 1s/epoch - 6ms/step
    Epoch 5/10
    194/194 - 1s - loss: 0.5676 - accuracy: 0.7812 - val_loss: 0.5352 - val_accuracy: 0.7904 - 1s/epoch - 5ms/step
    Epoch 6/10
    194/194 - 1s - loss: 0.5702 - accuracy: 0.7775 - val_loss: 0.5446 - val_accuracy: 0.7893 - 1s/epoch - 5ms/step
    Epoch 7/10
    194/194 - 1s - loss: 0.5627 - accuracy: 0.7844 - val_loss: 0.5374 - val_accuracy: 0.7876 - 1s/epoch - 6ms/step
    Epoch 8/10
    194/194 - 1s - loss: 0.5656 - accuracy: 0.7808 - val_loss: 0.5354 - val_accuracy: 0.7886 - 1s/epoch - 6ms/step
    Epoch 9/10
    194/194 - 1s - loss: 0.5627 - accuracy: 0.7831 - val_loss: 0.5426 - val_accuracy: 0.7846 - 1s/epoch - 6ms/step
    Epoch 10/10
    194/194 - 1s - loss: 0.5607 - accuracy: 0.7826 - val_loss: 0.5452 - val_accuracy: 0.7849 - 1s/epoch - 5ms/step
    Epoch 1/10
    194/194 - 1s - loss: 0.5651 - accuracy: 0.7828 - val_loss: 0.5077 - val_accuracy: 0.7971 - 1s/epoch - 6ms/step
    Epoch 2/10
    194/194 - 1s - loss: 0.5618 - accuracy: 0.7833 - val_loss: 0.5152 - val_accuracy: 0.7943 - 1s/epoch - 6ms/step
    Epoch 3/10
    194/194 - 1s - loss: 0.5627 - accuracy: 0.7833 - val_loss: 0.5333 - val_accuracy: 0.7876 - 1s/epoch - 6ms/step
    Epoch 4/10
    194/194 - 1s - loss: 0.5599 - accuracy: 0.7817 - val_loss: 0.5065 - val_accuracy: 0.7986 - 993ms/epoch - 5ms/step
    Epoch 5/10
    194/194 - 1s - loss: 0.5598 - accuracy: 0.7831 - val_loss: 0.5351 - val_accuracy: 0.7932 - 1s/epoch - 6ms/step
    Epoch 6/10
    194/194 - 1s - loss: 0.5624 - accuracy: 0.7816 - val_loss: 0.5143 - val_accuracy: 0.7969 - 1s/epoch - 6ms/step
    Epoch 7/10
    194/194 - 1s - loss: 0.5591 - accuracy: 0.7817 - val_loss: 0.5361 - val_accuracy: 0.7867 - 991ms/epoch - 5ms/step
    Epoch 8/10
    194/194 - 1s - loss: 0.5584 - accuracy: 0.7843 - val_loss: 0.5287 - val_accuracy: 0.7962 - 1s/epoch - 5ms/step
    Epoch 9/10
    194/194 - 1s - loss: 0.5567 - accuracy: 0.7849 - val_loss: 0.5195 - val_accuracy: 0.7954 - 1s/epoch - 6ms/step
    Epoch 10/10
    194/194 - 1s - loss: 0.5546 - accuracy: 0.7847 - val_loss: 0.5257 - val_accuracy: 0.7919 - 991ms/epoch - 5ms/step
    


```python
def make_submit_file(pred: np.ndarray, ss: pd.DataFrame) -> None:
    save_path = "submission.csv"
    submission.iloc[:, 1:] = pred
    submission.to_csv(save_path, index=None)
```


```python
make_submit_file(pred1, submission)
```


```python
!kaggle competitions submit -c otto-group-product-classification-challenge -f submission.csv -m "_1"
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.4)
    100% 27.2M/27.2M [00:02<00:00, 9.95MB/s]
    Successfully submitted to Otto Group Product Classification Challenge
