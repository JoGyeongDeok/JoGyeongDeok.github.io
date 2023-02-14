---
title: "PUBG Finish Placement Prediction (Kernels Only)"
tags: [Data Anaylsis, Code Review, Machine Learning]
comments: true
date : 2021-12-01
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

Kaggle PUBG Finish Placement Prediction (Kernels Only) 대회의 **ROB ROSE**의 코드를 참조하였다. feature마다 **메모리를 최소화** 하는 데이터 타입을 적용한 후 **PCA, K-means Clustering**을 통해 파생변수를 생성하였다. **다양한 방식으로 전처리**한 데이터(PCA, Clustering X)와 PCA, Clustering을 통해 얻은 파생변수 데이터들을 통해 Linear Regression을 진행하였고 가장 결과가 좋은 데이터를 최종 데이터로 사용하였다. 모델링은 **LGBMRegression**을 사용하였다.

<b><a href = 'https://www.kaggle.com/robroseknows/pubg-clustering-exploration'>코드 참조</a></b>

## 1.Library & Data Load


```python
from google.colab import drive
drive.mount('/content/drive')     
```

    Mounted at /content/drive
    


```python
!cp /content/drive/MyDrive/Kaggle/kaggle.json /root/.kaggle/
#!kaggle competitions list
!kaggle competitions download -c pubg-finish-placement-prediction
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.4)
    Downloading test_V2.csv.zip to /content
     97% 97.0M/100M [00:01<00:00, 53.5MB/s]
    100% 100M/100M [00:01<00:00, 74.0MB/s] 
    Downloading sample_submission_V2.csv.zip to /content
     29% 5.00M/17.0M [00:01<00:02, 5.11MB/s]
    100% 17.0M/17.0M [00:01<00:00, 15.9MB/s]
    Downloading train_V2.csv.zip to /content
     96% 235M/244M [00:05<00:00, 36.7MB/s]
    100% 244M/244M [00:05<00:00, 45.6MB/s]
    


```python
!unzip "train_V2.csv.zip" -d ""
!unzip "test_V2.csv.zip" -d ""
!unzip "sample_submission_V2.csv.zip" -d ""
```

    Archive:  train_V2.csv.zip
      inflating: train_V2.csv            
    Archive:  test_V2.csv.zip
      inflating: test_V2.csv             
    Archive:  sample_submission_V2.csv.zip
      inflating: sample_submission_V2.csv  
    
**변수 설명**
DBNOs - 적을 기절시킨 횟수

assists - 적을 죽이는데 도움을 준 횟수

boosts - 드링크, 진통제 사용 횟수

damageDealt - 딜량(적에게 준 총 피해량)(자해 데미지는 감산됨)

headshotKills - 헤드샷으로 적을 죽인 횟수

heals - 회복 아이템 사용 횟수

Id - 플레이어 ID

killPlace - Ranking in match of number of enemy players killed.

killPoints - 킬을 기반으로 한 레이팅(점수) (랭크포인트에 -1 이외의 값이 있는 경우 killPoints의 0은 "없음"으로 간주해야 합니다.)

killStreaks - 단 시간내에 사살한 적플레이어의 수

kills - 사살한 적 플레이어의 수

longestKill - 적을 사살했을 때 적 플레이어와 자신 사이의 거리(적을 사살하고 차를 몰고 떠나면 거리가 늘어나기 때문에 오차가 생길 수 있음)

matchDuration - 진행된 게임의 시간 (단위 - 초)

matchId - 매치를 구별하기 위한 매치ID

matchType - 게임의 유형(모드)“solo(1명)”, “duo(2명)”, “squad(4명)”, “solo-fpp(1명 1인칭)”, “duo-fpp(2명 2인칭)”, and “squad-fpp(4명 1인칭)”; 나머지는 이벤트 모드

rankPoints - 레이팅(점수) 랭킹을 나타내는 지표 -1값은 "없음"을 나타내는 값

revives - 이 플레이어가 팀을 부활시킨 횟수

rideDistance - 차량으로 이동한 총 거리 (단위 - 미터)

roadKills - 차량에 탑승한 동안 적을 죽인 횟수

swimDistance - 수영으로 이동한 총 거리 (단위 - 미터)

teamKills - 팀을 죽인 횟수

vehicleDestroys - 차량을 파괴한 횟수

walkDistance - 걸어서 다닌 거리 (단위 - 미터)

weaponsAcquired - 무기를 습득한 횟수

winPoints - 우승 기반 레이팅(점수) 적 사살횟수와 상관없는 승리 중심의 레이팅(rankPoints에 -1 이외의 값이 있는 경우 winPoints의 0은 "없음"으로 간주해야 합니다.)

groupId - 그룹을 식별하는 ID. 같은 그룹 이라도 다른 게임이라면 서로 다른 그룹아이디를 가짐. ex) 같은 듀오라도 첫판은 4d4b580de459be 두번째판은 684d5656442f9e 이런식으로 다른 그룹ID를 가짐

numGroups - 한 매치에 속한 그룹의 수(그룹ID의 수)

maxPlace - 꼴찌 팀의 순위(중간에 나가는 그룹이 있어서 numGroups 와 다를 수 있음)

winPlacePerc - 예측 타겟 데이터 백분위수를 기준으로 우승 순위를 배치 1은 1위에 해당하고 0은 꼴찌에 해당 maxPlace 를 기준으로 계산하므로 누락된 chunks 가 있을수 있다.


```python
import gc
import time
# Data
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import lightgbm as lgb
import random
random.seed(1337)
np.random.seed(1337)

import warnings
warnings.filterwarnings('ignore')

# winPlacePerc가 결측치인 id를 제외하고 불러오기
# Credit for this method here: https://www.kaggle.com/rejasupotaro/effective-feature-engineering
def reload():
    gc.collect()
    df = pd.read_csv('train_V2.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    return df
```


```python
df = reload()
```

## 2. Data 전처리

 - **iinfo와 finfo는 각각 int와 float의 표현가능한 수의 한계를 반환**




```python
np.iinfo(np.int8)
```




    iinfo(min=-128, max=127, dtype=int8)



**Memory saving function credit to**
 - https://www.kaggle.com/gemartin/load-data-reduce-memory-usage



```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            # numeric typedml 최대 최소
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
```

 - **MatchID 에 참여한 플레이어 수**


```python
df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
```

 - **게임마다 참여한 플레이어 수가 다르기 때문에 같은 Kills 이라도 각 MatchID에서의 정도가 다르다. 따라서 정규화가 필요하다.**
 - **playerJoined 에 관계없이 정규화**


```python
df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
df = reduce_mem_usage(df)
df.head()
```





  <div id="df-45727099-97aa-42aa-99f0-fe3dbbcfe2db">
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
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>longestKill</th>
      <th>matchDuration</th>
      <th>matchType</th>
      <th>maxPlace</th>
      <th>numGroups</th>
      <th>rankPoints</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7f96b2f878858a</td>
      <td>4d4b580de459be</td>
      <td>a10357fd1a4a91</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>1241</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1306</td>
      <td>squad-fpp</td>
      <td>28</td>
      <td>26</td>
      <td>-1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>244.75</td>
      <td>1</td>
      <td>1466</td>
      <td>0.444336</td>
      <td>96</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>29.12500</td>
      <td>1358.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>eef90569b9d03c</td>
      <td>684d5656442f9e</td>
      <td>aeb375fc57110c</td>
      <td>0</td>
      <td>0</td>
      <td>91.50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1777</td>
      <td>squad-fpp</td>
      <td>26</td>
      <td>25</td>
      <td>1484</td>
      <td>0</td>
      <td>0.004501</td>
      <td>0</td>
      <td>11.039062</td>
      <td>0</td>
      <td>0</td>
      <td>1434.00</td>
      <td>5</td>
      <td>0</td>
      <td>0.640137</td>
      <td>91</td>
      <td>0.000000</td>
      <td>99.6875</td>
      <td>28.34375</td>
      <td>1937.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1eaf90ac73de72</td>
      <td>6a4a42c3245a74</td>
      <td>110163d8bb94ae</td>
      <td>1</td>
      <td>0</td>
      <td>68.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1318</td>
      <td>duo</td>
      <td>50</td>
      <td>47</td>
      <td>1491</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>161.75</td>
      <td>2</td>
      <td>0</td>
      <td>0.775391</td>
      <td>98</td>
      <td>0.000000</td>
      <td>69.3750</td>
      <td>51.00000</td>
      <td>1344.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4616d365dd2853</td>
      <td>a930a9c79cd721</td>
      <td>f1f1f4ef412d7e</td>
      <td>0</td>
      <td>0</td>
      <td>32.90625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1436</td>
      <td>squad-fpp</td>
      <td>31</td>
      <td>30</td>
      <td>1408</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>202.75</td>
      <td>3</td>
      <td>0</td>
      <td>0.166748</td>
      <td>91</td>
      <td>0.000000</td>
      <td>35.8750</td>
      <td>33.78125</td>
      <td>1565.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>315c96c26c9aac</td>
      <td>de04010b3458dd</td>
      <td>6dc8ff871e21e6</td>
      <td>0</td>
      <td>0</td>
      <td>100.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>58.53125</td>
      <td>1424</td>
      <td>solo-fpp</td>
      <td>97</td>
      <td>95</td>
      <td>1560</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>49.75</td>
      <td>2</td>
      <td>0</td>
      <td>0.187500</td>
      <td>97</td>
      <td>1.030273</td>
      <td>103.0000</td>
      <td>99.93750</td>
      <td>1467.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-45727099-97aa-42aa-99f0-fe3dbbcfe2db')"
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
          document.querySelector('#df-45727099-97aa-42aa-99f0-fe3dbbcfe2db button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-45727099-97aa-42aa-99f0-fe3dbbcfe2db');
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




 - **Dropping columns with too many categories or unique values as well as the target column.**


```python
target = 'winPlacePerc'
drop_cols = ['Id', 'groupId', 'matchId', target]
select = [x for x in df.columns if x not in drop_cols]
X = df.loc[:, select]
X = pd.get_dummies(X)
X.head()
```





  <div id="df-59adec37-5395-4523-b417-8a51bf0c691d">
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
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>longestKill</th>
      <th>matchDuration</th>
      <th>maxPlace</th>
      <th>numGroups</th>
      <th>rankPoints</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>matchType_crashfpp</th>
      <th>matchType_crashtpp</th>
      <th>matchType_duo</th>
      <th>matchType_duo-fpp</th>
      <th>matchType_flarefpp</th>
      <th>matchType_flaretpp</th>
      <th>matchType_normal-duo</th>
      <th>matchType_normal-duo-fpp</th>
      <th>matchType_normal-solo</th>
      <th>matchType_normal-solo-fpp</th>
      <th>matchType_normal-squad</th>
      <th>matchType_normal-squad-fpp</th>
      <th>matchType_solo</th>
      <th>matchType_solo-fpp</th>
      <th>matchType_squad</th>
      <th>matchType_squad-fpp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>1241</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1306</td>
      <td>28</td>
      <td>26</td>
      <td>-1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>244.75</td>
      <td>1</td>
      <td>1466</td>
      <td>96</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>29.12500</td>
      <td>1358.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>91.50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1777</td>
      <td>26</td>
      <td>25</td>
      <td>1484</td>
      <td>0</td>
      <td>0.004501</td>
      <td>0</td>
      <td>11.039062</td>
      <td>0</td>
      <td>0</td>
      <td>1434.00</td>
      <td>5</td>
      <td>0</td>
      <td>91</td>
      <td>0.000000</td>
      <td>99.6875</td>
      <td>28.34375</td>
      <td>1937.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>68.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1318</td>
      <td>50</td>
      <td>47</td>
      <td>1491</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>161.75</td>
      <td>2</td>
      <td>0</td>
      <td>98</td>
      <td>0.000000</td>
      <td>69.3750</td>
      <td>51.00000</td>
      <td>1344.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>32.90625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1436</td>
      <td>31</td>
      <td>30</td>
      <td>1408</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>202.75</td>
      <td>3</td>
      <td>0</td>
      <td>91</td>
      <td>0.000000</td>
      <td>35.8750</td>
      <td>33.78125</td>
      <td>1565.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>100.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>58.53125</td>
      <td>1424</td>
      <td>97</td>
      <td>95</td>
      <td>1560</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>49.75</td>
      <td>2</td>
      <td>0</td>
      <td>97</td>
      <td>1.030273</td>
      <td>103.0000</td>
      <td>99.93750</td>
      <td>1467.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-59adec37-5395-4523-b417-8a51bf0c691d')"
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
          document.querySelector('#df-59adec37-5395-4523-b417-8a51bf0c691d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-59adec37-5395-4523-b417-8a51bf0c691d');
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




 - **PCA**


```python
pca2 = PCA(n_components=2)
pca2.fit(X)

print(sum(pca2.explained_variance_ratio_))
P2 = pca2.transform(X)
```

    0.7490323717751535
    


```python
pd.DataFrame(P2)
```





  <div id="df-997ce901-2940-41d9-89e7-3aec211661d8">
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1064.311035</td>
      <td>-1357.967163</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-335.500610</td>
      <td>984.300293</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-964.956055</td>
      <td>1070.167114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-929.486450</td>
      <td>1023.731995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-994.333191</td>
      <td>1124.211060</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4446960</th>
      <td>509.258667</td>
      <td>-1344.936279</td>
    </tr>
    <tr>
      <th>4446961</th>
      <td>-980.920410</td>
      <td>1088.687866</td>
    </tr>
    <tr>
      <th>4446962</th>
      <td>-689.465332</td>
      <td>1026.420410</td>
    </tr>
    <tr>
      <th>4446963</th>
      <td>187.467163</td>
      <td>823.278564</td>
    </tr>
    <tr>
      <th>4446964</th>
      <td>805.092896</td>
      <td>1024.664062</td>
    </tr>
  </tbody>
</table>
<p>4446965 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-997ce901-2940-41d9-89e7-3aec211661d8')"
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
          document.querySelector('#df-997ce901-2940-41d9-89e7-3aec211661d8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-997ce901-2940-41d9-89e7-3aec211661d8');
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





```python
plt.scatter(P2[:100000, 0], P2[:100000, 1])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_23_0.png)
    



```python
pca3 = PCA(n_components=3)
pca3.fit(X)
print(sum(pca3.explained_variance_ratio_))
P3 = pca3.transform(X)
```

    0.9608551926301087
    


```python
pd.DataFrame(P3)
```





  <div id="df-406a378b-e44e-478d-9fe4-984ad27a60a5">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1064.314209</td>
      <td>-1357.967529</td>
      <td>-630.891174</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-335.497467</td>
      <td>984.300476</td>
      <td>562.642517</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-964.958984</td>
      <td>1070.166748</td>
      <td>-547.455872</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-929.489319</td>
      <td>1023.731506</td>
      <td>-526.640869</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-994.336670</td>
      <td>1124.210938</td>
      <td>-645.543213</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4446960</th>
      <td>509.255859</td>
      <td>-1344.936401</td>
      <td>-538.462585</td>
    </tr>
    <tr>
      <th>4446961</th>
      <td>-980.923706</td>
      <td>1088.687500</td>
      <td>-628.636353</td>
    </tr>
    <tr>
      <th>4446962</th>
      <td>-689.465393</td>
      <td>1026.420410</td>
      <td>9.744699</td>
    </tr>
    <tr>
      <th>4446963</th>
      <td>187.476349</td>
      <td>823.279358</td>
      <td>1762.551880</td>
    </tr>
    <tr>
      <th>4446964</th>
      <td>805.091675</td>
      <td>1024.664062</td>
      <td>-184.807587</td>
    </tr>
  </tbody>
</table>
<p>4446965 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-406a378b-e44e-478d-9fe4-984ad27a60a5')"
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
          document.querySelector('#df-406a378b-e44e-478d-9fe4-984ad27a60a5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-406a378b-e44e-478d-9fe4-984ad27a60a5');
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





```python
fig_p3 = plt.figure()
ax = Axes3D(fig_p3, elev=48, azim=134)
ax.scatter(P3[:100000, 0], P3[:100000, 1], P3[:100000, 2])
fig_p3.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_26_0.png)
    


K-means


```python
kms = KMeans(n_clusters=2).fit(P2)
```


```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms.labels_[:100000])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_29_0.png)
    



```python
kms3 = KMeans(n_clusters=3).fit(P2)
kms4 = KMeans(n_clusters=4).fit(P2)
kms5 = KMeans(n_clusters=5).fit(P2)
```


```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms3.labels_[:100000])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_31_0.png)
    



```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms4.labels_[:100000])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_32_0.png)
    



```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms5.labels_[:100000])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_33_0.png)
    



```python
kms6 = KMeans(n_clusters=6).fit(P2)
kms7 = KMeans(n_clusters=7).fit(P2)
kms8 = KMeans(n_clusters=8).fit(P2)
```


```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms6.labels_[:100000])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_35_0.png)
    



```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms7.labels_[:100000])
plt.show()
```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_36_0.png)
    



```python
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms8.labels_[:100000])
plt.show()

```


    
![png](/images/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_files/2021-12-31-PUBG_Finish_Placement_Prediction_%28Kernels_Only%29_37_0.png)
    


 - **Feature Generation from K-Means**


```python
#PCA 후 kmeans로 군집화
def cluster_features(df, model, pca):
    P = pca.transform(df)
    new_df = pd.DataFrame()
    new_df['cluster'] = model.predict(P)
    one_hot = pd.get_dummies(new_df['cluster'], prefix='cluster')
    new_df = new_df.join(one_hot)
    new_df = new_df.drop('cluster', axis=1)
    new_df = new_df.fillna(0)
    return new_df
    
#PCA 후  각 군집의 중심으로 부터 거리 변수
def centroid_features(df, model, pca):
    P = pd.DataFrame(pca.transform(df))
    new_df = pd.DataFrame()
    cluster = 0
    for centers in model.cluster_centers_:
        new_df['distance_{}'.format(cluster)] = np.linalg.norm(P[[0, 1]].sub(np.array(centers)), axis=1)
        cluster += 1
    return new_df
```


```python
#정규화 함수
def norm_features(df):
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
    df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
    df = reduce_mem_usage(df)
    return df

#원핫 인코딩
def one_hot_encode(df):
    return pd.get_dummies(df, columns=['matchType'])

#쓸모없는 변수 제거
def remove_categories(df):
    target = 'winPlacePerc'
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', target]
    select = [x for x in df.columns if x not in drop_cols]
    return df.loc[:, select]
```


```python
def kmeans_5_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms5, pca2))
    
def kmeans_5_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms5, pca2))

def kmeans_3_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms3, pca2))
    
def kmeans_3_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms3, pca2))

def kmeans_4_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms4, pca2))
    
def kmeans_4_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms4, pca2))
```


```python
def train_test_split(df, test_size=0.1):
    match_ids = df['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    train = df[df['matchId'].isin(train_match_ids)]
    test = df[-df['matchId'].isin(train_match_ids)]
    
    return train, test
```

 - **전처리 후 LinearRegression() 후 에러 반환**


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def run_experiment(preprocess):
    df = reload()    

    df = preprocess(df)
    df.fillna(0, inplace=True)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    train, val = train_test_split(df, 0.1)
    
    model = LinearRegression()
    model.fit(train[cols_to_fit], train[target])
    
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_experiments(preprocesses):
    results = []
    for preprocess in preprocesses:
        start = time.time()
        score = run_experiment(preprocess)
        execution_time = time.time() - start
        results.append({
            'name': preprocess.__name__,
            'score': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['name', 'score', 'execution time']).sort_values(by='score')
```


```python
def original(df):
    return df

def items(df):
    df['items'] = df['heals'] + df['boosts']
    return df

def players_in_team(df):
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    return df.merge(agg, how='left', on=['groupId'])

def total_distance(df):
    df['total_distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']
    return df

def headshotKills_over_kills(df):
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['headshotKills_over_kills'].fillna(0, inplace=True)
    return df

def killPlace_over_maxPlace(df):
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['killPlace_over_maxPlace'].fillna(0, inplace=True)
    df['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)
    return df

def walkDistance_over_heals(df):
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_heals'].fillna(0, inplace=True)
    df['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)
    return df

def walkDistance_over_kills(df):
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['walkDistance_over_kills'].fillna(0, inplace=True)
    df['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)
    return df

def teamwork(df):
    df['teamwork'] = df['assists'] + df['revives']
    return df

def min_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId','groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def sum_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])

def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
```

**다양한 방식으로 전처리 후 Linear Regression 점수 반환**


```python
run_experiments([
    original,
    items,
    players_in_team,
    total_distance,
    headshotKills_over_kills,
    killPlace_over_maxPlace,
    walkDistance_over_heals,
    walkDistance_over_kills,
    teamwork
])
```





  <div id="df-e197420d-a750-4f48-8fcc-97f8206e612c">
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
      <th>name</th>
      <th>score</th>
      <th>execution time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>headshotKills_over_kills</td>
      <td>0.091615</td>
      <td>18.76s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>players_in_team</td>
      <td>0.091666</td>
      <td>37.76s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>items</td>
      <td>0.092305</td>
      <td>19.09s</td>
    </tr>
    <tr>
      <th>5</th>
      <td>killPlace_over_maxPlace</td>
      <td>0.092340</td>
      <td>18.48s</td>
    </tr>
    <tr>
      <th>0</th>
      <td>original</td>
      <td>0.092491</td>
      <td>21.03s</td>
    </tr>
    <tr>
      <th>8</th>
      <td>teamwork</td>
      <td>0.092519</td>
      <td>18.43s</td>
    </tr>
    <tr>
      <th>6</th>
      <td>walkDistance_over_heals</td>
      <td>0.092661</td>
      <td>19.04s</td>
    </tr>
    <tr>
      <th>3</th>
      <td>total_distance</td>
      <td>0.092883</td>
      <td>19.24s</td>
    </tr>
    <tr>
      <th>7</th>
      <td>walkDistance_over_kills</td>
      <td>0.093260</td>
      <td>18.3s</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e197420d-a750-4f48-8fcc-97f8206e612c')"
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
          document.querySelector('#df-e197420d-a750-4f48-8fcc-97f8206e612c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e197420d-a750-4f48-8fcc-97f8206e612c');
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





```python
# 각 군집으로 묶은 후 Linear Regression 점수 반환
run_experiments([
    original,
    kmeans_3_clusters, 
    kmeans_3_centroids,
    kmeans_4_clusters, 
    kmeans_4_centroids,
    kmeans_5_clusters, 
    kmeans_5_centroids
])
```





  <div id="df-b8fa5850-b5a8-4408-a9e4-0788a98b6bf8">
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
      <th>name</th>
      <th>score</th>
      <th>execution time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>kmeans_4_centroids</td>
      <td>0.089272</td>
      <td>26.84s</td>
    </tr>
    <tr>
      <th>5</th>
      <td>kmeans_5_clusters</td>
      <td>0.089349</td>
      <td>27.68s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kmeans_3_clusters</td>
      <td>0.089359</td>
      <td>27.08s</td>
    </tr>
    <tr>
      <th>6</th>
      <td>kmeans_5_centroids</td>
      <td>0.089414</td>
      <td>27.16s</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kmeans_4_clusters</td>
      <td>0.089562</td>
      <td>27.71s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kmeans_3_centroids</td>
      <td>0.089677</td>
      <td>25.15s</td>
    </tr>
    <tr>
      <th>0</th>
      <td>original</td>
      <td>0.092813</td>
      <td>18.79s</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b8fa5850-b5a8-4408-a9e4-0788a98b6bf8')"
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
          document.querySelector('#df-b8fa5850-b5a8-4408-a9e4-0788a98b6bf8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b8fa5850-b5a8-4408-a9e4-0788a98b6bf8');
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


```python
train = reload()
test = pd.read_csv('test_V2.csv')
submission = pd.read_csv('sample_submission_V2.csv')

train = kmeans_4_centroids(train)
test = kmeans_4_centroids(test)
```


```python
train.fillna(0, inplace = True)
test.fillna(0, inplace = True)
target = 'winPlacePerc'
cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', target]
cols_to_fit = [col for col in df.columns if col not in cols_to_drop]

```


```python
train_Y = train[target]
train_X = train[cols_to_fit]
test_X = test[cols_to_fit]
```


```python
train_X.head()
```





  <div id="df-fe2d4001-afb9-40b3-b4e3-f655a374ed93">
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
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>longestKill</th>
      <th>matchDuration</th>
      <th>maxPlace</th>
      <th>numGroups</th>
      <th>rankPoints</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>1241</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1306</td>
      <td>28</td>
      <td>26</td>
      <td>-1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>244.75</td>
      <td>1</td>
      <td>1466</td>
      <td>96</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>29.12500</td>
      <td>1358.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>91.50000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1777</td>
      <td>26</td>
      <td>25</td>
      <td>1484</td>
      <td>0</td>
      <td>0.004501</td>
      <td>0</td>
      <td>11.039062</td>
      <td>0</td>
      <td>0</td>
      <td>1434.00</td>
      <td>5</td>
      <td>0</td>
      <td>91</td>
      <td>0.000000</td>
      <td>99.6875</td>
      <td>28.34375</td>
      <td>1937.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>68.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1318</td>
      <td>50</td>
      <td>47</td>
      <td>1491</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>161.75</td>
      <td>2</td>
      <td>0</td>
      <td>98</td>
      <td>0.000000</td>
      <td>69.3750</td>
      <td>51.00000</td>
      <td>1344.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>32.90625</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1436</td>
      <td>31</td>
      <td>30</td>
      <td>1408</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>202.75</td>
      <td>3</td>
      <td>0</td>
      <td>91</td>
      <td>0.000000</td>
      <td>35.8750</td>
      <td>33.78125</td>
      <td>1565.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>100.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>58.53125</td>
      <td>1424</td>
      <td>97</td>
      <td>95</td>
      <td>1560</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>49.75</td>
      <td>2</td>
      <td>0</td>
      <td>97</td>
      <td>1.030273</td>
      <td>103.0000</td>
      <td>99.93750</td>
      <td>1467.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fe2d4001-afb9-40b3-b4e3-f655a374ed93')"
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
          document.querySelector('#df-fe2d4001-afb9-40b3-b4e3-f655a374ed93 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fe2d4001-afb9-40b3-b4e3-f655a374ed93');
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




## 3. 모델링


```python
from lightgbm import LGBMRegressor
```


```python
lgbm = LGBMRegressor(random_state = 0, n_estimators = 100)
lgbm.fit(train_X,train_Y)
pred = lgbm.predict(test_X)
```


```python
submission.winPlacePerc = pred
```


```python
!pip install kaggle==1.5.12
```

    Requirement already satisfied: kaggle==1.5.12 in /usr/local/lib/python3.7/dist-packages (1.5.12)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (1.24.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (2.23.0)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (2.8.2)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (2021.10.8)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (1.15.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (4.62.3)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle==1.5.12) (5.0.2)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle==1.5.12) (1.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle==1.5.12) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle==1.5.12) (2.10)
    


```python
submission
```





  <div id="df-98ef0249-d988-4bb1-93ae-d45b107c04f7">
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
      <th>Id</th>
      <th>winPlacePerc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9329eb41e215eb</td>
      <td>0.224336</td>
    </tr>
    <tr>
      <th>1</th>
      <td>639bd0dcd7bda8</td>
      <td>0.938831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>63d5c8ef8dfe91</td>
      <td>0.639555</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cf5b81422591d1</td>
      <td>0.513089</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ee6a295187ba21</td>
      <td>0.928715</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1934169</th>
      <td>a316c3a13887d5</td>
      <td>0.736692</td>
    </tr>
    <tr>
      <th>1934170</th>
      <td>5312146b27d875</td>
      <td>0.395652</td>
    </tr>
    <tr>
      <th>1934171</th>
      <td>fc8818b5b32ad3</td>
      <td>0.876815</td>
    </tr>
    <tr>
      <th>1934172</th>
      <td>a0f91e35f8458f</td>
      <td>0.823465</td>
    </tr>
    <tr>
      <th>1934173</th>
      <td>3696fc9f3a42b2</td>
      <td>0.047315</td>
    </tr>
  </tbody>
</table>
<p>1934174 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-98ef0249-d988-4bb1-93ae-d45b107c04f7')"
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
          document.querySelector('#df-98ef0249-d988-4bb1-93ae-d45b107c04f7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-98ef0249-d988-4bb1-93ae-d45b107c04f7');
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
