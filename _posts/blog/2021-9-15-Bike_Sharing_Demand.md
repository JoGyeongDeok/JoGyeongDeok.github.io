---
title: "Bike Sharing Demand"
category: Kaggle
tags: [Data Anaylsis, Code Review, Machine Learning]
comments: true
date : 2021-09-15
categories: 
  - blog
excerpt: Kaggle 코드 리뷰
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
---

<br><br>
Kaggle Bike Sharing Demand 대회의 **Mohamed El-Kasaby**의 코드를 참조하였다. 온도, 시간대 데이터를 통해 파생변수를 생성하고 **히스토그램 기반 그레디언트 부스팅 기법**을 사용하였다.

<b><a href = 'https://www.kaggle.com/mohamedalkasaby/b-s-d-solution
'>코드 참조</a></b>


# 1.Library & Data Load


```python
from google.colab import drive
drive.mount('/content/drive')     
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
!cp /content/drive/MyDrive/Kaggle/kaggle.json /root/.kaggle/
#!kaggle competitions list
!kaggle competitions download -c bike-sharing-demand
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.4)
    Downloading test.csv to /content
      0% 0.00/316k [00:00<?, ?B/s]
    100% 316k/316k [00:00<00:00, 29.0MB/s]
    Downloading sampleSubmission.csv to /content
      0% 0.00/140k [00:00<?, ?B/s]
    100% 140k/140k [00:00<00:00, 43.6MB/s]
    Downloading train.csv to /content
      0% 0.00/633k [00:00<?, ?B/s]
    100% 633k/633k [00:00<00:00, 86.6MB/s]
    


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split
%matplotlib inline
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import MaxAbsScaler,PowerTransformer,MinMaxScaler,RobustScaler

from xgboost import XGBRegressor

from scipy import stats

import warnings
warnings.filterwarnings('ignore')
```


```python
train_df= pd.read_csv('train.csv')
test_df= pd.read_csv('test.csv')
date=test_df.datetime
submission = pd.read_csv("sampleSubmission.csv")
```


```python
# parse datetime colum & add new time related columns
train_df['datetime']=pd.to_datetime(train_df['datetime'])

train_df['day'] = train_df['datetime'].dt.day_name()
train_df['month'] = train_df['datetime'].dt.month_name()
train_df['year'] = train_df['datetime'].dt.year
train_df['hour'] = train_df['datetime'].dt.hour
train_df['dayofweek'] = train_df['datetime'].dt.dayofweek
train_df['weekofyear'] = train_df['datetime'].dt.weekofyear

train_df = train_df.drop('datetime',axis=1)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.
      if __name__ == '__main__':
    


```python
display(train_df.describe().T)
```



  <div id="df-0f0ca5a0-d3e6-4a52-ae27-2899ab1be920">
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>season</th>
      <td>10886.0</td>
      <td>2.506614</td>
      <td>1.116174</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>3.000</td>
      <td>4.0000</td>
      <td>4.0000</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>10886.0</td>
      <td>0.028569</td>
      <td>0.166599</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>10886.0</td>
      <td>0.680875</td>
      <td>0.466159</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>weather</th>
      <td>10886.0</td>
      <td>1.418427</td>
      <td>0.633839</td>
      <td>1.00</td>
      <td>1.0000</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>4.0000</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>10886.0</td>
      <td>20.230860</td>
      <td>7.791590</td>
      <td>0.82</td>
      <td>13.9400</td>
      <td>20.500</td>
      <td>26.2400</td>
      <td>41.0000</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>10886.0</td>
      <td>23.655084</td>
      <td>8.474601</td>
      <td>0.76</td>
      <td>16.6650</td>
      <td>24.240</td>
      <td>31.0600</td>
      <td>45.4550</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>10886.0</td>
      <td>61.886460</td>
      <td>19.245033</td>
      <td>0.00</td>
      <td>47.0000</td>
      <td>62.000</td>
      <td>77.0000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>10886.0</td>
      <td>12.799395</td>
      <td>8.164537</td>
      <td>0.00</td>
      <td>7.0015</td>
      <td>12.998</td>
      <td>16.9979</td>
      <td>56.9969</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>10886.0</td>
      <td>36.021955</td>
      <td>49.960477</td>
      <td>0.00</td>
      <td>4.0000</td>
      <td>17.000</td>
      <td>49.0000</td>
      <td>367.0000</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>10886.0</td>
      <td>155.552177</td>
      <td>151.039033</td>
      <td>0.00</td>
      <td>36.0000</td>
      <td>118.000</td>
      <td>222.0000</td>
      <td>886.0000</td>
    </tr>
    <tr>
      <th>count</th>
      <td>10886.0</td>
      <td>191.574132</td>
      <td>181.144454</td>
      <td>1.00</td>
      <td>42.0000</td>
      <td>145.000</td>
      <td>284.0000</td>
      <td>977.0000</td>
    </tr>
    <tr>
      <th>year</th>
      <td>10886.0</td>
      <td>2011.501929</td>
      <td>0.500019</td>
      <td>2011.00</td>
      <td>2011.0000</td>
      <td>2012.000</td>
      <td>2012.0000</td>
      <td>2012.0000</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>10886.0</td>
      <td>11.541613</td>
      <td>6.915838</td>
      <td>0.00</td>
      <td>6.0000</td>
      <td>12.000</td>
      <td>18.0000</td>
      <td>23.0000</td>
    </tr>
    <tr>
      <th>dayofweek</th>
      <td>10886.0</td>
      <td>3.013963</td>
      <td>2.004585</td>
      <td>0.00</td>
      <td>1.0000</td>
      <td>3.000</td>
      <td>5.0000</td>
      <td>6.0000</td>
    </tr>
    <tr>
      <th>weekofyear</th>
      <td>10886.0</td>
      <td>25.917784</td>
      <td>15.017269</td>
      <td>1.00</td>
      <td>14.0000</td>
      <td>26.000</td>
      <td>40.0000</td>
      <td>52.0000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0f0ca5a0-d3e6-4a52-ae27-2899ab1be920')"
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
      align : center;
      max-height:500px;
      max-width:100%;
      overflow : auto;
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
          document.querySelector('#df-0f0ca5a0-d3e6-4a52-ae27-2899ab1be920 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0f0ca5a0-d3e6-4a52-ae27-2899ab1be920');
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
display(train_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 17 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   season      10886 non-null  int64  
     1   holiday     10886 non-null  int64  
     2   workingday  10886 non-null  int64  
     3   weather     10886 non-null  int64  
     4   temp        10886 non-null  float64
     5   atemp       10886 non-null  float64
     6   humidity    10886 non-null  int64  
     7   windspeed   10886 non-null  float64
     8   casual      10886 non-null  int64  
     9   registered  10886 non-null  int64  
     10  count       10886 non-null  int64  
     11  day         10886 non-null  object 
     12  month       10886 non-null  object 
     13  year        10886 non-null  int64  
     14  hour        10886 non-null  int64  
     15  dayofweek   10886 non-null  int64  
     16  weekofyear  10886 non-null  int64  
    dtypes: float64(3), int64(12), object(2)
    memory usage: 1.4+ MB
    


    None



```python
sns.pairplot(train_df[["season","weather","temp","atemp","humidity","windspeed","count"]], diag_kind="kde")
```




    <seaborn.axisgrid.PairGrid at 0x7f08e57650d0>




    
![png](/images/2021-12-31-Bike_Sharing_Demand_files/2021-12-31-Bike_Sharing_Demand_10_1.png)
    



```python
train_df.weather.value_counts()
```




    1    7192
    2    2834
    3     859
    4       1
    Name: weather, dtype: int64



# 2. Data 전처리

 - <b>duplicates Checking</b>


```python
train_df.duplicated().sum()
```




    0




```python
weather_df = train_df[["season","weather"]] 
date_df =train_df [['day','month','year','hour','dayofweek','weekofyear']]
```


```python
j = 0
plt.figure(figsize=(25, 25))
for i in weather_df: 
    j=j+1
    plt.subplot(7, 4, j+1)
    sns.barplot(weather_df[i],train_df["count"])
    plt.show
```


    
![png](/images/2021-12-31-Bike_Sharing_Demand_files/2021-12-31-Bike_Sharing_Demand_16_0.png)
    



```python
train_df.loc[train_df.weather==4,:]
```





  <div id="df-faa01d0e-2a21-4577-8c22-dee656317066">
    <div class=".colab-df-container">
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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5631</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>8.2</td>
      <td>11.365</td>
      <td>86</td>
      <td>6.0032</td>
      <td>6</td>
      <td>158</td>
      <td>164</td>
      <td>Monday</td>
      <td>January</td>
      <td>2012</td>
      <td>18</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-faa01d0e-2a21-4577-8c22-dee656317066')"
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
          document.querySelector('#df-faa01d0e-2a21-4577-8c22-dee656317066 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-faa01d0e-2a21-4577-8c22-dee656317066');
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
j = 0
plt.figure(figsize=(25, 10))
for i in date_df: 
    j=j+1
    plt.subplot(2, 4, j+1)
    sns.barplot(date_df[i],train_df["count"])
    plt.show
```


    
![png](/images/2021-12-31-Bike_Sharing_Demand_files/2021-12-31-Bike_Sharing_Demand_18_0.png)
    



```python
train_df["temp"].describe()
```




    count    10886.00000
    mean        20.23086
    std          7.79159
    min          0.82000
    25%         13.94000
    50%         20.50000
    75%         26.24000
    max         41.00000
    Name: temp, dtype: float64

<br><br>

 - <b>온도 범주화</b>


```python
train_df["temp_range"] = train_df["temp"]
for indx, i in enumerate(list(train_df["temp"])):
    if i <=10:
        train_df["temp_range"][indx] = 0
    elif 10 < i <= 18 :
        train_df["temp_range"][indx] = 2
    elif 18 < i <= 25 :
        train_df["temp_range"][indx] = 3
    elif 25 < i <= 32 :
        train_df["temp_range"][indx] = 4
    elif 32 < i :
        train_df["temp_range"][indx] = 1
```


```python
train_df[['count','temp_range']].groupby(train_df["temp_range"]).sum()
```





  <div id="df-1ddfca93-ae4c-40c3-b0fa-e2ba8c163452">
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
      <th>count</th>
      <th>temp_range</th>
    </tr>
    <tr>
      <th>temp_range</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>92141</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>208573</td>
      <td>605.0</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>460170</td>
      <td>6290.0</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>585791</td>
      <td>8952.0</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>738801</td>
      <td>11572.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1ddfca93-ae4c-40c3-b0fa-e2ba8c163452')"
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
          document.querySelector('#df-1ddfca93-ae4c-40c3-b0fa-e2ba8c163452 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1ddfca93-ae4c-40c3-b0fa-e2ba8c163452');
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



<br><br>

 - <b>파생변수 생성</b>


```python
train_df['RushHour']= train_df['hour'].isin([8,17,18,19,20,21])
train_df['lowHour']= train_df['hour'].isin([0,1,2,3,4])
train_df['DayorNight'] = (train_df['hour'] >= 7) & (train_df['hour'] <= 20)
```


```python
display(train_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 21 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   season      10886 non-null  int64  
     1   holiday     10886 non-null  int64  
     2   workingday  10886 non-null  int64  
     3   weather     10886 non-null  int64  
     4   temp        10886 non-null  float64
     5   atemp       10886 non-null  float64
     6   humidity    10886 non-null  int64  
     7   windspeed   10886 non-null  float64
     8   casual      10886 non-null  int64  
     9   registered  10886 non-null  int64  
     10  count       10886 non-null  int64  
     11  day         10886 non-null  object 
     12  month       10886 non-null  object 
     13  year        10886 non-null  int64  
     14  hour        10886 non-null  int64  
     15  dayofweek   10886 non-null  int64  
     16  weekofyear  10886 non-null  int64  
     17  temp_range  10886 non-null  float64
     18  RushHour    10886 non-null  bool   
     19  lowHour     10886 non-null  bool   
     20  DayorNight  10886 non-null  bool   
    dtypes: bool(3), float64(4), int64(12), object(2)
    memory usage: 1.5+ MB
    


    None



```python
train_df["day"] = train_df["day"].astype('category').cat.codes
train_df["month"] = train_df["month"].astype('category').cat.codes
```


```python
train_df.head(10)
```





  <div id="df-b525a97d-c8cd-4c6d-a9ab-bd8a9636a381">
    <div class=".colab-df-container">
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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>weekofyear</th>
      <th>temp_range</th>
      <th>RushHour</th>
      <th>lowHour</th>
      <th>DayorNight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0000</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>0</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0000</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>1</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0000</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>2</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0000</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>3</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>4</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>9.84</td>
      <td>12.880</td>
      <td>75</td>
      <td>6.0032</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>5</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0000</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>6</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8.20</td>
      <td>12.880</td>
      <td>86</td>
      <td>0.0000</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>7</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0000</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>8</td>
      <td>5</td>
      <td>52</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13.12</td>
      <td>17.425</td>
      <td>76</td>
      <td>0.0000</td>
      <td>8</td>
      <td>6</td>
      <td>14</td>
      <td>2</td>
      <td>4</td>
      <td>2011</td>
      <td>9</td>
      <td>5</td>
      <td>52</td>
      <td>2.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b525a97d-c8cd-4c6d-a9ab-bd8a9636a381')"
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
          document.querySelector('#df-b525a97d-c8cd-4c6d-a9ab-bd8a9636a381 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b525a97d-c8cd-4c6d-a9ab-bd8a9636a381');
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
Y  = train_df['count']
Y1 = train_df['casual']
Y2 = train_df['registered']
```


```python
train_df = train_df.drop(["year","atemp","count","casual","registered"],axis = 1)
```

<br><br>

 - <b>시간대 구분 및 변수 생성</b>



```python
test_df['datetime']=pd.to_datetime(test_df['datetime'])
test_df['day'] = test_df['datetime'].dt.day_name()
test_df['month'] = test_df['datetime'].dt.month_name()
test_df['year'] = test_df['datetime'].dt.year
test_df['hour'] = test_df['datetime'].dt.hour
test_df['dayofweek'] = test_df['datetime'].dt.dayofweek
test_df['weekofyear'] = test_df['datetime'].dt.weekofyear
test_df = test_df.drop('datetime',axis=1)
############################################################
test_df["temp_range"] = test_df["temp"]
for indx, i in enumerate(list(test_df["temp"])):
    if i <=10:
        test_df["temp_range"][indx] = 0
    elif 10 < i <= 18 :
        test_df["temp_range"][indx] = 2
    elif 18 < i <= 25 :
        test_df["temp_range"][indx] = 3
    elif 25 < i <= 32 :
        test_df["temp_range"][indx] = 4
    elif 32 < i :
        test_df["temp_range"][indx] = 1
############################################################
# some new column
test_df['RushHour']= test_df['hour'].isin([8,17,18,19,20,21])
test_df['lowHour']= test_df['hour'].isin([0,1,2,3,4])
test_df['DayorNight'] = (test_df['hour'] >= 7) & (test_df['hour'] <= 20)
############################################################
test_df["day"] = test_df["day"].astype('category').cat.codes
test_df["month"] = test_df["month"].astype('category').cat.codes
############################################################
test_df = test_df.drop(["year","atemp"],axis = 1)
```
<br><br>

# 3. 모델링


```python
x_train, x_val, y_train, y_val = train_test_split(train_df , Y , test_size = 0.05, random_state = 29)
```


```python
#evaluation matrix
from math import sqrt
def rmsle(y_pred , y_actual):
    n = y_pred.size 
    RMSLE = sqrt(((np.log(y_pred+1)-np.log(y_actual+1))**2).sum()/n)
    return RMSLE

from sklearn.metrics import make_scorer
myScorer = make_scorer(rmsle, greater_is_better=False)

```

 - <b>히스토그램 기반 그레디언트 부스팅</b>

      히스토그램 기반 그레디언트 부스팅(Histogram-based Gradient Boosting)은 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘이다. 히스토그램 기반 그레디언트 부스팅은 먼저 입력 특성을 256개의 구간으로 나눈다. 따라서 노드를 분할할 때 최적의 분할을 매우 빠르게 찾을 수 있다.

      또한 256개의 구간 중에서 하나를 떼어 놓고 누락된 값을 위해서 사용한다. 따라서 입력에 누락된 특성이 있더라도 이를 따로 전처리할 필요가 없다.

      사이킷런의 히스토그램 기반 그레이디언트 부스팅 클래스는 HistGradientBoostingClassifier이다. 일반적으로 기본 매개변수에서 안정적인 성능을 얻을 수 있다. 트리의 개수를 지정하는데 n_estimators 대신에 부스팅 반복 횟수를 지정하는 max_iter를 사용한다.

      하지만  이 알고리즘은 아직  실험적인 단계라서 인터페이스가 노출되어 있지 않다.


      LGBM은 저수준 언어로 되어있어 빠르지만, HistGBM은 파이썬 언어로 되어있어 느리다.

<br>


```python
HistGradient = HistGradientBoostingRegressor()

param = {#n_estimators' : [180], 
    'max_iter':[115],
    'max_depth' : [11],
    'max_leaf_nodes':[15],
    'max_bins':[150]
         #min_samples_split':[2],
         #min_samples_leaf':[1],
        }
gridSearch_HistGradient = GridSearchCV(HistGradient,param,scoring=myScorer,cv=10,verbose=3)
gridSearch_HistGradient.fit(x_train,np.log1p(y_train))

best_HistGradient = gridSearch_HistGradient.best_estimator_
bestHistGradient_testScore=best_HistGradient.score(x_train, np.log1p(y_train))

```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    [CV 1/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.084 total time=   0.4s
    [CV 2/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.091 total time=   0.3s
    [CV 3/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.088 total time=   0.4s
    [CV 4/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.087 total time=   0.7s
    [CV 5/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.093 total time=   0.3s
    [CV 6/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.085 total time=   0.3s
    [CV 7/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.088 total time=   0.3s
    [CV 8/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.089 total time=   0.4s
    [CV 9/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.094 total time=   0.3s
    [CV 10/10] END max_bins=150, max_depth=11, max_iter=115, max_leaf_nodes=15;, score=-0.090 total time=   0.3s
    


```python
gridSearch_HistGradient.best_params_
```




    {'max_bins': 150, 'max_depth': 11, 'max_iter': 115, 'max_leaf_nodes': 15}




```python
bestHistGradient_testScore
```




    0.9419974204247508




```python
pred=np.expm1(best_HistGradient.predict(x_val))
```


```python
print(rmsle(pred,y_val))
```

    0.39485737137746585
    


```python
predictions = np.expm1(best_HistGradient.predict(test_df))
```


```python
predictions = pd.DataFrame({'datetime':date,
                       'count': predictions})
```


```python
predictions.to_csv("final_submission.csv", index=False)
!kaggle competitions submit -c bike-sharing-demand -f final_submission.csv -m "_1"
```