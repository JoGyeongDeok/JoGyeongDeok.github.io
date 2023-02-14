---
title: "New York City Taxi Trip Durationd"
tags: [Data Anaylsis, Code Review, Machine Learning]
comments: true
date : 2021-10-15
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
Kaggle New York City Taxi Trip Durationd 대회의 **YAGANA SHERIFF-HUSSAINI**의 코드를 참조하였다. 운행 기간을 정렬하여 산점도를 그린 후 산점도가 끊어진 위치에서 이상치를 제거하였다. datetime을 통해 여러가지 파생변수를 생성하였고, 위도 경도 데이터를 통해 **PCA**와 **Clustering** 기법을 통해 파생변수를 생성하였다. 이후 **TPOT** 모듈을 사용해 모델링 하였다. 

<b><a href = 'https://www.kaggle.com/sheriytm/brewed-tpot-for-nyc-with-love-lb0-37
'>코드 참조</a></b>


## 1.Library & Data Load


```python
from google.colab import drive
drive.mount('/content/drive')     
```

    Mounted at /content/drive
    


```python
!cp /content/drive/MyDrive/Kaggle/kaggle.json /root/.kaggle/
#!kaggle competitions list
!kaggle competitions download -c nyc-taxi-trip-duration
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.4)
    Downloading train.zip to /content
     86% 54.0M/62.9M [00:00<00:00, 88.4MB/s]
    100% 62.9M/62.9M [00:00<00:00, 83.6MB/s]
    Downloading test.zip to /content
     93% 19.0M/20.3M [00:01<00:00, 15.6MB/s]
    100% 20.3M/20.3M [00:01<00:00, 18.3MB/s]
    Downloading sample_submission.zip to /content
      0% 0.00/2.49M [00:00<?, ?B/s]
    100% 2.49M/2.49M [00:00<00:00, 136MB/s]
    


```python
!unzip "/content/train.zip" -d "/content"
!unzip "/content/test.zip" -d "/content"
!unzip "/content/sample_submission.zip" -d "/content"
```

    Archive:  /content/train.zip
      inflating: /content/train.csv      
    Archive:  /content/test.zip
      inflating: /content/test.csv       
    Archive:  /content/sample_submission.zip
      inflating: /content/sample_submission.csv  
    


```python
!pip install haversine
!pip install tpot
```

    Collecting haversine
      Downloading haversine-2.5.1-py2.py3-none-any.whl (6.1 kB)
    Installing collected packages: haversine
    Successfully installed haversine-2.5.1
    Collecting tpot
      Downloading TPOT-0.11.7-py3-none-any.whl (87 kB)
    [K     |████████████████████████████████| 87 kB 4.7 MB/s 
    [?25hCollecting deap>=1.2
      Downloading deap-1.3.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (160 kB)
    [K     |████████████████████████████████| 160 kB 43.4 MB/s 
    [?25hRequirement already satisfied: tqdm>=4.36.1 in /usr/local/lib/python3.7/dist-packages (from tpot) (4.62.3)
    Requirement already satisfied: scikit-learn>=0.22.0 in /usr/local/lib/python3.7/dist-packages (from tpot) (1.0.1)
    Requirement already satisfied: numpy>=1.16.3 in /usr/local/lib/python3.7/dist-packages (from tpot) (1.19.5)
    Requirement already satisfied: pandas>=0.24.2 in /usr/local/lib/python3.7/dist-packages (from tpot) (1.1.5)
    Collecting stopit>=1.1.1
      Downloading stopit-1.1.2.tar.gz (18 kB)
    Collecting update-checker>=0.16
      Downloading update_checker-0.18.0-py3-none-any.whl (7.0 kB)
    Requirement already satisfied: joblib>=0.13.2 in /usr/local/lib/python3.7/dist-packages (from tpot) (1.1.0)
    Requirement already satisfied: scipy>=1.3.1 in /usr/local/lib/python3.7/dist-packages (from tpot) (1.4.1)
    Collecting xgboost>=1.1.0
      Downloading xgboost-1.5.1-py3-none-manylinux2014_x86_64.whl (173.5 MB)
    [K     |████████████████████████████████| 173.5 MB 10 kB/s 
    [?25hRequirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.2->tpot) (2.8.2)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.2->tpot) (2018.9)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas>=0.24.2->tpot) (1.15.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.22.0->tpot) (3.0.0)
    Requirement already satisfied: requests>=2.3.0 in /usr/local/lib/python3.7/dist-packages (from update-checker>=0.16->tpot) (2.23.0)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.3.0->update-checker>=0.16->tpot) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.3.0->update-checker>=0.16->tpot) (2021.10.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.3.0->update-checker>=0.16->tpot) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.3.0->update-checker>=0.16->tpot) (1.24.3)
    Building wheels for collected packages: stopit
      Building wheel for stopit (setup.py) ... [?25l[?25hdone
      Created wheel for stopit: filename=stopit-1.1.2-py3-none-any.whl size=11952 sha256=f690cc9eae2afbfa2f0bb798244477c69867cc2d4c4889834e96cb8ffb05ac92
      Stored in directory: /root/.cache/pip/wheels/e2/d2/79/eaf81edb391e27c87f51b8ef901ecc85a5363dc96b8b8d71e3
    Successfully built stopit
    Installing collected packages: xgboost, update-checker, stopit, deap, tpot
      Attempting uninstall: xgboost
        Found existing installation: xgboost 0.90
        Uninstalling xgboost-0.90:
          Successfully uninstalled xgboost-0.90
    Successfully installed deap-1.3.1 stopit-1.1.2 tpot-0.11.7 update-checker-0.18.0 xgboost-1.5.1
    


```python
# Import libraries
import os
mingw_path = 'g:/mingw64/bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
from haversine import haversine
import datetime as dt

from subprocess import check_output

import warnings
warnings.filterwarnings('ignore')
```


```python
os.listdir()
```




    ['.config',
     'train.zip',
     'sample_submission.csv',
     'train.csv',
     'test.csv',
     'sample_submission.zip',
     'drive',
     'test.zip',
     'sample_data']




```python
trainDF = pd.read_csv('train.csv', nrows=50000)

# Load testing data as test
testDF = pd.read_csv('test.csv')

# Print size as well as the top 5 observation of training dataset
print('Size of the training set is: {} rows and {} columns'.format(*trainDF.shape))
print ("\n", trainDF.head(5))
```

    Size of the training set is: 50000 rows and 11 columns
    
               id  vendor_id  ... store_and_fwd_flag trip_duration
    0  id2875421          2  ...                  N           455
    1  id2377394          1  ...                  N           663
    2  id3858529          2  ...                  N          2124
    3  id3504673          2  ...                  N           429
    4  id2181028          2  ...                  N           435
    
    [5 rows x 11 columns]
    


```python
trainDF.describe()
```





  <div id="df-adfc4f3c-f0bf-4709-868c-36187ab4f543">
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
      <th>vendor_id</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.529440</td>
      <td>1.667260</td>
      <td>-73.973477</td>
      <td>40.751211</td>
      <td>-73.973374</td>
      <td>40.752025</td>
      <td>949.708280</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.499138</td>
      <td>1.316442</td>
      <td>0.037974</td>
      <td>0.027994</td>
      <td>0.036672</td>
      <td>0.032536</td>
      <td>3175.391374</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-74.393555</td>
      <td>40.449749</td>
      <td>-74.398514</td>
      <td>40.444698</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-73.991760</td>
      <td>40.737579</td>
      <td>-73.991348</td>
      <td>40.736086</td>
      <td>395.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>-73.981644</td>
      <td>40.754425</td>
      <td>-73.979698</td>
      <td>40.754610</td>
      <td>659.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>-73.967079</td>
      <td>40.768513</td>
      <td>-73.962868</td>
      <td>40.769985</td>
      <td>1071.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>-73.518051</td>
      <td>41.091171</td>
      <td>-72.711395</td>
      <td>41.311520</td>
      <td>86357.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-adfc4f3c-f0bf-4709-868c-36187ab4f543')"
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
      margin-left: auto;
      margin-right: auto;
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
          document.querySelector('#df-adfc4f3c-f0bf-4709-868c-36187ab4f543 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-adfc4f3c-f0bf-4709-868c-36187ab4f543');
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
print("Train shape : ", trainDF.shape)
print("Test shape : ", testDF.shape)
```

    Train shape :  (50000, 11)
    Test shape :  (625134, 9)
    


```python
dtypeDF = trainDF.dtypes.reset_index()
dtypeDF.columns = ["Count", "Column Type"]
dtypeDF.groupby("Column Type").aggregate('count').reset_index()
```





  <div id="df-b46acb85-a149-43c0-b2cc-9e17f37fa166">
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
      <th>Column Type</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int64</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>float64</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>object</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b46acb85-a149-43c0-b2cc-9e17f37fa166')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>


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
          document.querySelector('#df-b46acb85-a149-43c0-b2cc-9e17f37fa166 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b46acb85-a149-43c0-b2cc-9e17f37fa166');
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




 - **Plot the trip duration**


```python
plt.figure(figsize=(8,6))
plt.scatter(range(trainDF.shape[0]), np.sort(trainDF.trip_duration.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('trip duration', fontsize=12)
plt.show()
```


    
![png](/images/2021-12-31-New_York_City_Taxi_Trip_Duration_files/2021-12-31-New_York_City_Taxi_Trip_Duration_14_0.png)
    



```python
th = trainDF.trip_duration.quantile(0.99)
tempDF = trainDF
tempDF = tempDF[tempDF['trip_duration'] < th]
plt.figure(figsize=(8,6))
plt.scatter(range(tempDF.shape[0]), np.sort(tempDF.trip_duration.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('trip duration', fontsize=12)
plt.show()

del tempDF
```


    
![png](/images/2021-12-31-New_York_City_Taxi_Trip_Duration_files/2021-12-31-New_York_City_Taxi_Trip_Duration_15_0.png)
    


Lets remove the outliers from the train data target


```python
trainDF = trainDF[trainDF['trip_duration'] < th]
```

Number of variables with missing values


```python
variables_missing_value = trainDF.isnull().sum()
variables_missing_value 
```




    id                    0
    vendor_id             0
    pickup_datetime       0
    dropoff_datetime      0
    passenger_count       0
    pickup_longitude      0
    pickup_latitude       0
    dropoff_longitude     0
    dropoff_latitude      0
    store_and_fwd_flag    0
    trip_duration         0
    dtype: int64




```python
variables_missing_value = testDF.isnull().sum()
variables_missing_value 
```




    id                    0
    vendor_id             0
    pickup_datetime       0
    passenger_count       0
    pickup_longitude      0
    pickup_latitude       0
    dropoff_longitude     0
    dropoff_latitude      0
    store_and_fwd_flag    0
    dtype: int64



## 2. Data 전처리


```python
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

t0 = dt.datetime.now()

train = trainDF
test = testDF

#pickup, dropoff datetime 변환
# 탑승한 날짜 변수 추가
# train 에 check_trip_duration: 탑승시각-하차시각 변수 추가
#주성분분석을 통해 변수 추가
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)


train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).map(lambda x: x.total_seconds())

duration_difference = train[np.abs(train['check_trip_duration'].values  - train['trip_duration'].values) > 1]
print('Trip_duration and datetimes are ok.') if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0 else print('Ooops.')

train['trip_duration'].describe()

train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

# Feature Extraction
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
```

    Trip_duration and datetimes are ok.
    
<img src = 'https://drive.google.com/uc?id=1vI8AYgJ-lfH89f7GArnx7rgq9W3LHgFf' height = 1000 width = 700 align = "center">

**Distance**

 - 탄 위치의 거리 + 직선거리, 맨해튼거리


```python

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])


train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

# Datetime features 
#요일
#1년중 몇째주인지
# 일주일을 시간으로계산

train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear
train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']

test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear
test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']

train.loc[:,'week_delta'] = train['pickup_datetime'].dt.weekday + \
    ((train['pickup_datetime'].dt.hour + (train['pickup_datetime'].dt.minute / 60.0)) / 24.0)
test.loc[:,'week_delta'] = test['pickup_datetime'].dt.weekday + \
    ((test['pickup_datetime'].dt.hour + (test['pickup_datetime'].dt.minute / 60.0)) / 24.0)

# Make time features cyclic
train.loc[:,'week_delta_sin'] = np.sin((train['week_delta'] / 7) * np.pi)**2
train.loc[:,'hour_sin'] = np.sin((train['pickup_hour'] / 24) * np.pi)**2

test.loc[:,'week_delta_sin'] = np.sin((test['week_delta'] / 7) * np.pi)**2
test.loc[:,'hour_sin'] = np.sin((test['pickup_hour'] / 24) * np.pi)**2


# Speed
train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)


# Average speed for regions
gby_cols = ['pickup_lat_bin', 'pickup_long_bin']
coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
coord_stats = coord_stats[coord_stats['id'] > 100]

train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)
train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)
train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)
test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)
test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))
```

 - **Clustering**


```python
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
t1 = dt.datetime.now()
print('Time till clustering: %i seconds' % (t1 - t0).seconds)
```

    Time till clustering: 85 seconds
    


```python
# Temporal and geospatial aggregation
for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',
               'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']:
    gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']]
    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

for gby_cols in [['center_lat_bin', 'center_long_bin'],
                 ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                 ['pickup_hour', 'pickup_cluster'],  ['pickup_hour', 'dropoff_cluster'],
                 ['pickup_cluster', 'dropoff_cluster']]:
    coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
    coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
    coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
    coord_stats = coord_stats[coord_stats['id'] > 100]
    coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
    train = pd.merge(train, coord_stats, how='left', on=gby_cols)
    test = pd.merge(test, coord_stats, how='left', on=gby_cols)

group_freq = '60min'
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

# Count trips over 60min
df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
train = train.merge(df_counts, on='id', how='left')
test = test.merge(df_counts, on='id', how='left')
```


```python
feature_names = list(train.columns)
print(np.setdiff1d(train.columns, test.columns))
do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'center_long_bin', 'pickup_dt_bin', 'pickup_datetime_group']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
# print(feature_names)
print('We have %i features.' % len(feature_names))
train[feature_names].count()
#ytrain = np.log(train['trip_duration'].values + 1)
#ytrain = train[train['trip_duration'].notnull()]['trip_duration_log'].values


t1 = dt.datetime.now()
print('Feature extraction time: %i seconds' % (t1 - t0).seconds)
```

    ['avg_speed_h' 'avg_speed_m' 'check_trip_duration' 'dropoff_datetime'
     'log_trip_duration' 'trip_duration']
    We have 57 features.
    Feature extraction time: 92 seconds
    


```python
feature_stats = pd.DataFrame({'feature': feature_names})
feature_stats.loc[:, 'train_mean'] = np.nanmean(train[feature_names].values, axis=0).round(4)
feature_stats.loc[:, 'test_mean'] = np.nanmean(test[feature_names].values, axis=0).round(4)
feature_stats.loc[:, 'train_std'] = np.nanstd(train[feature_names].values, axis=0).round(4)
feature_stats.loc[:, 'test_std'] = np.nanstd(test[feature_names].values, axis=0).round(4)
feature_stats.loc[:, 'train_nan'] = np.mean(np.isnan(train[feature_names].values), axis=0).round(3)
feature_stats.loc[:, 'test_nan'] = np.mean(np.isnan(test[feature_names].values), axis=0).round(3)
feature_stats.loc[:, 'train_test_mean_diff'] = np.abs(feature_stats['train_mean'] - feature_stats['test_mean']) / np.abs(feature_stats['train_std'] + feature_stats['test_std'])  * 2
feature_stats.loc[:, 'train_test_nan_diff'] = np.abs(feature_stats['train_nan'] - feature_stats['test_nan'])
feature_stats = feature_stats.sort_values(by='train_test_mean_diff')
feature_stats[['feature', 'train_test_mean_diff']].tail()
```





  <div id="df-4e0004cf-e9df-49ea-9bda-1a40ee588f10">
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
      <th>feature</th>
      <th>train_test_mean_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>pca_manhattan</td>
      <td>0.033898</td>
    </tr>
    <tr>
      <th>52</th>
      <td>avg_speed_h_pickup_hour_dropoff_cluster</td>
      <td>0.036043</td>
    </tr>
    <tr>
      <th>53</th>
      <td>cnt_pickup_hour_dropoff_cluster</td>
      <td>0.070988</td>
    </tr>
    <tr>
      <th>50</th>
      <td>avg_speed_h_pickup_hour_pickup_cluster</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>51</th>
      <td>cnt_pickup_hour_pickup_cluster</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4e0004cf-e9df-49ea-9bda-1a40ee588f10')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>

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
          document.querySelector('#df-4e0004cf-e9df-49ea-9bda-1a40ee588f10 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4e0004cf-e9df-49ea-9bda-1a40ee588f10');
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

 **TPOT**
 - TPOT는 유전 프로그래밍을 사용하여 기계 학습 파이프라인을 최적화하는 Python Automated Machine Learning 도구입니다.
 - TPOT는 Scikit-learn을 기반으로 만들어졌으므로 TPOT가 생성하는 모든 코드가 익숙해 보일 것입니다.
 - TPOT는 아직 개발 중이므로 정기적으로 이 저장소를 확인하여 업데이트를 받으시기 바랍니다.


```python
from tpot import TPOTRegressor
auto_classifier = TPOTRegressor(generations=3, population_size=9, verbosity=2)
from sklearn.model_selection import train_test_split
```


```python
# K Fold Cross Validation
from sklearn.model_selection import KFold

X = train[feature_names].values
y = np.log(train['trip_duration'].values + 1)  


kf = KFold(n_splits=10)
kf.get_n_splits(X)

print(kf)  

KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
```

    KFold(n_splits=10, random_state=None, shuffle=False)
    TRAIN: [ 4950  4951  4952 ... 49497 49498 49499] TEST: [   0    1    2 ... 4947 4948 4949]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [4950 4951 4952 ... 9897 9898 9899]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [ 9900  9901  9902 ... 14847 14848 14849]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [14850 14851 14852 ... 19797 19798 19799]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [19800 19801 19802 ... 24747 24748 24749]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [24750 24751 24752 ... 29697 29698 29699]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [29700 29701 29702 ... 34647 34648 34649]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [34650 34651 34652 ... 39597 39598 39599]
    TRAIN: [    0     1     2 ... 49497 49498 49499] TEST: [39600 39601 39602 ... 44547 44548 44549]
    TRAIN: [    0     1     2 ... 44547 44548 44549] TEST: [44550 44551 44552 ... 49497 49498 49499]
    


```python
auto_classifier.fit(X_train, y_train)
```


```python
# Now do the prediction
test_result = auto_classifier.predict(test[feature_names].values)
sub = pd.DataFrame()
sub['id'] = test['id']
sub['trip_duration'] = np.exp(test_result)
sub.to_csv('NYCTaxi_TpotModels.csv', index=False)
!kaggle competitions submit -c nyc-taxi-trip-duration -f NYCTaxi_TpotModels.csv -m "_1"
```

    Successfully submitted to New York City Taxi Trip Duration
