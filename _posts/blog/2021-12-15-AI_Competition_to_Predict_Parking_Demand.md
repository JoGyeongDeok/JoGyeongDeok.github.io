---
title: "AI Competition to Predict Parking Demand"
tags: [Data Anaylsis, Machine Learning]
comments: true
date : 2021-12-15
categories: 
  - blog
excerpt: Dacon Contest
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---


# 1.Library & Data Load


```python
from google.colab import drive
drive.mount('/content/drive')      
```


```python
# !mkdir "/content/drive/MyDrive/DACON/Data/AI Competition to Predict Parking Demand"
# !mkdir "/content/drive/MyDrive/DACON/Data/AI Competition to Predict Parking Demand/submission"
# !unzip "/content/drive/MyDrive/DACON/Data/AI Competition to Predict Parking Demand.zip" -d "/content/drive/MyDrive/DACON/Data/AI Competition to Predict Parking Demand/"
```

    mkdir: cannot create directory ‘/content/drive/MyDrive/DACON/Data/Computer Vision Outlier Detection Algorithm Contest’: No such file or directory
    mkdir: cannot create directory ‘/content/drive/MyDrive/DACON/Data/Computer Vision Outlier Detection Algorithm Contest/submission’: No such file or directory
    unzip:  cannot find or open /content/drive/MyDrive/DACON/Data/Computer Vision Outlier Detection Algorithm Contest.zip, /content/drive/MyDrive/DACON/Data/Computer Vision Outlier Detection Algorithm Contest.zip.zip or /content/drive/MyDrive/DACON/Data/Computer Vision Outlier Detection Algorithm Contest.zip.ZIP.
    


```python
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf

plt.rc('font', family='NanumBarunGothic') 
%matplotlib inline
plt.rcParams['axes.unicode_minus'] = False # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

```

## 데이터 불러오기


```python
path='/content/drive/MyDrive/DACON/Data/AI Competition to Predict Parking Demand'
```


```python
train = pd.read_csv(path+'/train.csv')
test = pd.read_csv(path+'/test.csv')
submission= pd.read_csv(path+'/sample_submission.csv')
age_gender_info=pd.read_csv(path+'/age_gender_info.csv')
```


```python
#데이터 잘못된거 1번해결 전용면적별 세대수 다 더함
for code in tqdm(train.단지코드.unique()):
  train.loc[train[train.단지코드==code].index,('총세대수')]=train.loc[train[train.단지코드==code].index,('전용면적별세대수')].sum()
  
for code in tqdm(test.단지코드.unique()):
  test.loc[test[test.단지코드==code].index,('총세대수')]=test.loc[test[test.단지코드==code].index,('전용면적별세대수')].sum()  
```


      0%|          | 0/423 [00:00<?, ?it/s]



      0%|          | 0/150 [00:00<?, ?it/s]



```python
#데이터 잘못된거 2번해결
train.loc[train[train.단지코드=='C1397'].index,('단지코드')]='C2085'
train.loc[train[train.단지코드=='C2085'].index,('총세대수')]=train.loc[train[train.단지코드=='C2085'].index,('총세대수')].unique().sum()

train.loc[train[train.단지코드=='C1649'].index,('단지코드')]='C2431'
train.loc[train[train.단지코드=='C2431'].index,('총세대수')]=train.loc[train[train.단지코드=='C2431'].index,('총세대수')].unique().sum()
train.loc[train[train.단지코드=='C2431'].index,('등록차량수')]=train.loc[train[train.단지코드=='C2431'].index,('등록차량수')].unique().sum()

test.loc[test[test.단지코드=='C1036'].index,('단지코드')]='C2675'
test.loc[test[test.단지코드=='C2675'].index,('총세대수')]=1254
```


```python
train.loc[train.임대료=='-'].단지코드.unique()
```




    array(['C2085', 'C1039', 'C1326', 'C1786', 'C2186'], dtype=object)




```python
test[test.임대료=='-'].단지코드.unique()
```




    array(['C2152', 'C1267'], dtype=object)




```python
#데이터 잘못된거 3번해결
test[test.임대료=='-'].단지코드.unique()
drop_list=['C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988','C2085']
for code in drop_list:
  _index=train[train.단지코드==code].index
  train=train.drop(_index)
```


```python
data=pd.concat([train,test]).reset_index(drop=True)
```


```python
data.loc[data.임대보증금=='-','임대보증금']=np.nan
data.loc[data.임대료=='-','임대료']=np.nan
data.임대보증금=data.임대보증금.fillna(-1)
data.임대료=data.임대료.fillna(-1)
data=data.astype({'임대보증금':'int', '임대료':'int'})
data.loc[data.임대료==-1,'임대료']=np.nan
data.loc[data.임대보증금==-1,'임대보증금']=np.nan
```


```python
data.isnull().sum()
```




    단지코드                               0
    총세대수                               0
    임대건물구분                             0
    지역                                 0
    공급유형                               0
    전용면적                               0
    전용면적별세대수                           0
    공가수                                0
    자격유형                               2
    임대보증금                            767
    임대료                              770
    도보 10분거리 내 지하철역 수(환승노선 수 반영)     253
    도보 10분거리 내 버스정류장 수                 4
    단지내주차면수                            0
    등록차량수                           1022
    dtype: int64




```python
data.describe()
```




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
      <th>총세대수</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>도보 10분거리 내 지하철역 수(환승노선 수 반영)</th>
      <th>도보 10분거리 내 버스정류장 수</th>
      <th>단지내주차면수</th>
      <th>등록차량수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3904.000000</td>
      <td>3904.000000</td>
      <td>3904.000000</td>
      <td>3904.000000</td>
      <td>3.137000e+03</td>
      <td>3.134000e+03</td>
      <td>3651.000000</td>
      <td>3900.000000</td>
      <td>3904.000000</td>
      <td>2882.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>878.888576</td>
      <td>44.276217</td>
      <td>102.217982</td>
      <td>13.607838</td>
      <td>2.570353e+07</td>
      <td>1.879700e+05</td>
      <td>0.169269</td>
      <td>3.947692</td>
      <td>582.123975</td>
      <td>553.280014</td>
    </tr>
    <tr>
      <th>std</th>
      <td>524.188912</td>
      <td>33.083356</td>
      <td>131.274005</td>
      <td>10.826404</td>
      <td>1.876572e+07</td>
      <td>1.192975e+05</td>
      <td>0.433334</td>
      <td>3.621108</td>
      <td>379.840543</td>
      <td>432.045047</td>
    </tr>
    <tr>
      <th>min</th>
      <td>26.000000</td>
      <td>9.960000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.249000e+06</td>
      <td>1.665000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>499.000000</td>
      <td>32.100000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.441800e+07</td>
      <td>1.107900e+05</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>278.000000</td>
      <td>209.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>775.000000</td>
      <td>39.820000</td>
      <td>60.000000</td>
      <td>13.000000</td>
      <td>2.052500e+07</td>
      <td>1.572000e+05</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>490.000000</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1116.000000</td>
      <td>49.960000</td>
      <td>141.250000</td>
      <td>21.000000</td>
      <td>3.195800e+07</td>
      <td>2.319075e+05</td>
      <td>0.000000</td>
      <td>4.250000</td>
      <td>804.000000</td>
      <td>763.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2572.000000</td>
      <td>583.400000</td>
      <td>1865.000000</td>
      <td>55.000000</td>
      <td>2.138630e+08</td>
      <td>1.058030e+06</td>
      <td>3.000000</td>
      <td>50.000000</td>
      <td>1798.000000</td>
      <td>2550.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 함수


```python
def fillna_(df, na_columns, method):
  imp = SimpleImputer(missing_values = np.nan, strategy = method)
  for col in na_columns:
    df[col] = imp.fit_transform(df[[col]])
  return df
```


```python
def code_dupli(data,col):
  num =[]
  for code in data.단지코드:
    num.append(data.loc[data.단지코드==code,col].unique().shape[0])
  plt.hist(num)
  plt.title(col)

```

# 2. Data 전처리

## 이산형데이터 탐색


```python
sns.countplot(x="임대건물구분", data=data)
plt.title("임대건물구분")
plt.show()
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_22_0.png' height = '400' width = '600'>
    



```python
sns.countplot(x="공급유형", data=data)
plt.title("공급유형")
plt.show()
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_23_0.png' height = '400' width = '600'>
    



```python
sns.countplot(x="자격유형", data=data)
plt.title("자격유형")
plt.show()
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_24_0.png' height = '400' width = '600'>
    



```python
code_dupli(data,'임대건물구분')
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_25_0.png' height = '400' width = '600'>
    



```python
code_dupli(data,'지역')
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_26_0.png' height = '400' width = '600'>
    



```python
code_dupli(data,'공급유형')
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_27_0.png' height = '400' width = '600'>
    



```python
code_dupli(data,'자격유형')
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_28_0.png' height = '400' width = '600'>
    



```python
pd.crosstab(data['공급유형'], data['자격유형'], margins=True).style.background_gradient(cmap='summer_r')
```


  <div id="df-355c5119-c3e2-4538-bd96-baf381dde928">


<style  type="text/css" >
#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col15,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row3_col15,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col15,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row8_col15,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col8{
            background-color:  #ffff66;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row0_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col15{
            background-color:  #fdfe66;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col0{
            background-color:  #e8f466;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row1_col15{
            background-color:  #f2f866;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row2_col0{
            background-color:  #fbfd66;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row4_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col3{
            background-color:  #fefe66;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col0{
            background-color:  #1e8e66;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col0,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col1,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col2,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col3,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col4,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col5,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col6,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col7,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col8,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col9,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col10,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col11,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col12,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col13,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col14,#T_d001f204_5805_11ec_927b_0242ac1c0002row10_col15{
            background-color:  #008066;
            color:  #f1f1f1;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col4{
            background-color:  #108866;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row5_col15{
            background-color:  #65b266;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col4{
            background-color:  #eff766;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row6_col15{
            background-color:  #f3f966;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col3{
            background-color:  #038166;
            color:  #f1f1f1;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row7_col15{
            background-color:  #cfe766;
            color:  #000000;
        }#T_d001f204_5805_11ec_927b_0242ac1c0002row9_col15{
            background-color:  #eaf466;
            color:  #000000;
        }</style><table id="T_d001f204_5805_11ec_927b_0242ac1c0002" class=""><thead>    <tr>        <th class="index_name level0" >자격유형</th>        <th class="col_heading level0 col0" >A</th>        <th class="col_heading level0 col1" >B</th>        <th class="col_heading level0 col2" >C</th>        <th class="col_heading level0 col3" >D</th>        <th class="col_heading level0 col4" >E</th>        <th class="col_heading level0 col5" >F</th>        <th class="col_heading level0 col6" >G</th>        <th class="col_heading level0 col7" >H</th>        <th class="col_heading level0 col8" >I</th>        <th class="col_heading level0 col9" >J</th>        <th class="col_heading level0 col10" >K</th>        <th class="col_heading level0 col11" >L</th>        <th class="col_heading level0 col12" >M</th>        <th class="col_heading level0 col13" >N</th>        <th class="col_heading level0 col14" >O</th>        <th class="col_heading level0 col15" >All</th>    </tr>    <tr>        <th class="index_name level0" >공급유형</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row0" class="row_heading level0 row0" >공공분양</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col0" class="data row0 col0" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col1" class="data row0 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col2" class="data row0 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col3" class="data row0 col3" >7</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col4" class="data row0 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col5" class="data row0 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col6" class="data row0 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col7" class="data row0 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col8" class="data row0 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col9" class="data row0 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col10" class="data row0 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col11" class="data row0 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col12" class="data row0 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col13" class="data row0 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col14" class="data row0 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row0_col15" class="data row0 col15" >7</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row1" class="row_heading level0 row1" >공공임대(10년)</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col0" class="data row1 col0" >214</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col1" class="data row1 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col2" class="data row1 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col3" class="data row1 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col4" class="data row1 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col5" class="data row1 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col6" class="data row1 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col7" class="data row1 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col8" class="data row1 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col9" class="data row1 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col10" class="data row1 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col11" class="data row1 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col12" class="data row1 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col13" class="data row1 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col14" class="data row1 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row1_col15" class="data row1 col15" >214</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row2" class="row_heading level0 row2" >공공임대(50년)</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col0" class="data row2 col0" >44</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col1" class="data row2 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col2" class="data row2 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col3" class="data row2 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col4" class="data row2 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col5" class="data row2 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col6" class="data row2 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col7" class="data row2 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col8" class="data row2 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col9" class="data row2 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col10" class="data row2 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col11" class="data row2 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col12" class="data row2 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col13" class="data row2 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col14" class="data row2 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row2_col15" class="data row2 col15" >44</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row3" class="row_heading level0 row3" >공공임대(5년)</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col0" class="data row3 col0" >3</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col1" class="data row3 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col2" class="data row3 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col3" class="data row3 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col4" class="data row3 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col5" class="data row3 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col6" class="data row3 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col7" class="data row3 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col8" class="data row3 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col9" class="data row3 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col10" class="data row3 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col11" class="data row3 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col12" class="data row3 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col13" class="data row3 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col14" class="data row3 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row3_col15" class="data row3 col15" >3</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row4" class="row_heading level0 row4" >공공임대(분납)</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col0" class="data row4 col0" >13</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col1" class="data row4 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col2" class="data row4 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col3" class="data row4 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col4" class="data row4 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col5" class="data row4 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col6" class="data row4 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col7" class="data row4 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col8" class="data row4 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col9" class="data row4 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col10" class="data row4 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col11" class="data row4 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col12" class="data row4 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col13" class="data row4 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col14" class="data row4 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row4_col15" class="data row4 col15" >13</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row5" class="row_heading level0 row5" >국민임대</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col0" class="data row5 col0" >2035</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col1" class="data row5 col1" >21</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col2" class="data row5 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col3" class="data row5 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col4" class="data row5 col4" >44</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col5" class="data row5 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col6" class="data row5 col6" >10</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col7" class="data row5 col7" >247</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col8" class="data row5 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col9" class="data row5 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col10" class="data row5 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col11" class="data row5 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col12" class="data row5 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col13" class="data row5 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col14" class="data row5 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row5_col15" class="data row5 col15" >2357</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row6" class="row_heading level0 row6" >영구임대</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col0" class="data row6 col0" >2</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col1" class="data row6 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col2" class="data row6 col2" >129</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col3" class="data row6 col3" >3</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col4" class="data row6 col4" >3</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col5" class="data row6 col5" >3</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col6" class="data row6 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col7" class="data row6 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col8" class="data row6 col8" >56</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col9" class="data row6 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col10" class="data row6 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col11" class="data row6 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col12" class="data row6 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col13" class="data row6 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col14" class="data row6 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row6_col15" class="data row6 col15" >196</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row7" class="row_heading level0 row7" >임대상가</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col0" class="data row7 col0" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col1" class="data row7 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col2" class="data row7 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col3" class="data row7 col3" >739</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col4" class="data row7 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col5" class="data row7 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col6" class="data row7 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col7" class="data row7 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col8" class="data row7 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col9" class="data row7 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col10" class="data row7 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col11" class="data row7 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col12" class="data row7 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col13" class="data row7 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col14" class="data row7 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row7_col15" class="data row7 col15" >739</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row8" class="row_heading level0 row8" >장기전세</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col0" class="data row8 col0" >3</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col1" class="data row8 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col2" class="data row8 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col3" class="data row8 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col4" class="data row8 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col5" class="data row8 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col6" class="data row8 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col7" class="data row8 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col8" class="data row8 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col9" class="data row8 col9" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col10" class="data row8 col10" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col11" class="data row8 col11" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col12" class="data row8 col12" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col13" class="data row8 col13" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col14" class="data row8 col14" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row8_col15" class="data row8 col15" >3</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row9" class="row_heading level0 row9" >행복주택</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col0" class="data row9 col0" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col1" class="data row9 col1" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col2" class="data row9 col2" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col3" class="data row9 col3" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col4" class="data row9 col4" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col5" class="data row9 col5" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col6" class="data row9 col6" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col7" class="data row9 col7" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col8" class="data row9 col8" >0</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col9" class="data row9 col9" >187</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col10" class="data row9 col10" >49</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col11" class="data row9 col11" >45</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col12" class="data row9 col12" >4</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col13" class="data row9 col13" >40</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col14" class="data row9 col14" >1</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row9_col15" class="data row9 col15" >326</td>
            </tr>
            <tr>
                        <th id="T_d001f204_5805_11ec_927b_0242ac1c0002level0_row10" class="row_heading level0 row10" >All</th>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col0" class="data row10 col0" >2314</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col1" class="data row10 col1" >21</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col2" class="data row10 col2" >129</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col3" class="data row10 col3" >749</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col4" class="data row10 col4" >47</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col5" class="data row10 col5" >3</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col6" class="data row10 col6" >10</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col7" class="data row10 col7" >247</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col8" class="data row10 col8" >56</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col9" class="data row10 col9" >187</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col10" class="data row10 col10" >49</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col11" class="data row10 col11" >45</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col12" class="data row10 col12" >4</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col13" class="data row10 col13" >40</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col14" class="data row10 col14" >1</td>
                        <td id="T_d001f204_5805_11ec_927b_0242ac1c0002row10_col15" class="data row10 col15" >3902</td>
            </tr>
    </tbody></table>
    </div>



위의 그래프는 각 단지의 column별 종류의 개수를 히스토그램으로 나타낸 것이다.
임대건물구분, 공급유형, 자격유형은 하나의 단지의 여러 종류의 유형이 들어가 있음을 알 수 있다.
위에 크로스탭을 보아 자격유형 (J,K,L,M,N,O), (B,G,H), (C,F,I) 묶고
공급유형(공공임대(10년), 공공임대(50년), 공공임대(5년), 공공임대(분납), 장기전세), (임대상가, 공공분양) 이렇게 다시 범주화 한다.)



```python
data.loc[(data.자격유형=='J')|(data.자격유형=='K')|(data.자격유형=='L')|(data.자격유형=='M')|(data.자격유형=='N')|(data.자격유형=='O'),'자격유형']='JKLMNO'
data.loc[(data.자격유형=='B')|(data.자격유형=='G')|(data.자격유형=='H'),'자격유형']='BGH'
data.loc[(data.자격유형=='C')|(data.자격유형=='F')|(data.자격유형=='I'),'자격유형']='CFI'

data.loc[(data.공급유형=='공공임대(5년)')|(data.공급유형=='공공임대(10년)')|(data.공급유형=='공공임대(50년)')|(data.공급유형=='공공임대(분납)')|(data.공급유형=='장기전세'),'공급유형']='공공임대_장기전세'
data.loc[(data.공급유형=='임대상가')|(data.공급유형=='공공분양'),'공급유형']='임대상가_공공분양'
```

## 연속형 변수 탐색


```python
plt.boxplot(data[['전용면적']].dropna().values, notch=True)
plt.xticks([1],['전용면적'])
```




    ([<matplotlib.axis.XTick at 0x7f5b353f5ed0>], [Text(0, 0, '전용면적')])




    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_33_1.png' height = '400' width = '600'>
    



```python
plt.boxplot(data[['임대보증금']].dropna().values, notch=True)
plt.xticks([1],['임대보증금'])
```




    ([<matplotlib.axis.XTick at 0x7f5b353f5bd0>], [Text(0, 0, '임대보증금')])




    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_34_1.png' height = '400' width = '600'>
    



```python
plt.boxplot(data[['임대료']].dropna().values, notch=True)
plt.xticks([1],['임대료'])
```




    ([<matplotlib.axis.XTick at 0x7f5b364d6850>], [Text(0, 0, '임대료')])




    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_35_1.png' height = '400' width = '600'>
    



```python
data[['전용면적','임대보증금','임대료','도보 10분거리 내 지하철역 수(환승노선 수 반영)','도보 10분거리 내 버스정류장 수','단지내주차면수']].describe()
```




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
      <th>전용면적</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>도보 10분거리 내 지하철역 수(환승노선 수 반영)</th>
      <th>도보 10분거리 내 버스정류장 수</th>
      <th>단지내주차면수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3904.000000</td>
      <td>3.137000e+03</td>
      <td>3.134000e+03</td>
      <td>3651.000000</td>
      <td>3900.000000</td>
      <td>3904.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>44.276217</td>
      <td>2.570353e+07</td>
      <td>1.879700e+05</td>
      <td>0.169269</td>
      <td>3.947692</td>
      <td>582.123975</td>
    </tr>
    <tr>
      <th>std</th>
      <td>33.083356</td>
      <td>1.876572e+07</td>
      <td>1.192975e+05</td>
      <td>0.433334</td>
      <td>3.621108</td>
      <td>379.840543</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.960000</td>
      <td>2.249000e+06</td>
      <td>1.665000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.100000</td>
      <td>1.441800e+07</td>
      <td>1.107900e+05</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.820000</td>
      <td>2.052500e+07</td>
      <td>1.572000e+05</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>490.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.960000</td>
      <td>3.195800e+07</td>
      <td>2.319075e+05</td>
      <td>0.000000</td>
      <td>4.250000</td>
      <td>804.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>583.400000</td>
      <td>2.138630e+08</td>
      <td>1.058030e+06</td>
      <td>3.000000</td>
      <td>50.000000</td>
      <td>1798.000000</td>
    </tr>
  </tbody>
</table>
</div>





## 이상치 처리


```python
#이상치 제거
data.loc[data.전용면적>200,'전용면적']=200
data.loc[data.임대보증금>150000000,'임대보증금']=150000000
data.loc[data.임대료>800000,'임대료']=800000
```




```python
#지하철 변수를 지하철의 유 무
#버스 정류장 개수를 1, 2, 3, 4, 5 이상으로 재범주화

data.loc[data['도보 10분거리 내 지하철역 수(환승노선 수 반영)']==0,'도보 10분거리 내 지하철역 수(환승노선 수 반영)']='무'
data.loc[data['도보 10분거리 내 지하철역 수(환승노선 수 반영)']!='무','도보 10분거리 내 지하철역 수(환승노선 수 반영)']='유'
data.loc[data['도보 10분거리 내 버스정류장 수']>=5,'도보 10분거리 내 버스정류장 수']='5 이상'
```


```python
# LabelEncoder를 이용하여 지역, 지하철, 버스 라벨 인코더

data['지역'] = LabelEncoder().fit_transform(data['지역'])
data['도보 10분거리 내 지하철역 수(환승노선 수 반영)'] = LabelEncoder().fit_transform(data['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].astype(str))
data['도보 10분거리 내 버스정류장 수'] = LabelEncoder().fit_transform(data['도보 10분거리 내 버스정류장 수'].astype(str))
```

## 결측치 처리


```python
data.isnull().sum()
```




    단지코드                               0
    총세대수                               0
    임대건물구분                             0
    지역                                 0
    공급유형                               0
    전용면적                               0
    전용면적별세대수                           0
    공가수                                0
    자격유형                               2
    임대보증금                            767
    임대료                              770
    도보 10분거리 내 지하철역 수(환승노선 수 반영)       0
    도보 10분거리 내 버스정류장 수                 0
    단지내주차면수                            0
    등록차량수                           1022
    dtype: int64



data의 결측값을 확인했더니  임대보증금, 임대료, 도보 10분거리 내 지하철역 수(환승노선 수 반영), 도보 10분거리 내 버스정류장 수 변수에 결측값이 존재한다


```python
#임대보증금, 임대료 결측치 처리
data.loc[(data.자격유형!='D')&(data.임대료.isna())].단지코드.unique()
```




    array(['C1039', 'C1326', 'C1786', 'C2186', 'C2152', 'C1267'], dtype=object)




```python
na_rent=data.loc[data.임대료.isna()]
na_security_deposit=data.loc[data.임대보증금.isna()]
```


```python
for i in na_rent.index:
  data.loc[i,'임대료'] = data.loc[(data.임대건물구분==na_rent.임대건물구분[i]) & (data.공급유형==na_rent.공급유형[i]) & (data.자격유형==na_rent.자격유형[i]) & (data.지역==na_rent.지역[i]) & (data.전용면적>=na_rent.전용면적[i]-5) & (data.전용면적<=na_rent.전용면적[i]+5)].임대료.mean()

for i in na_security_deposit.index:
  data.loc[i,'임대보증금']  = data.loc[(data.임대건물구분==na_security_deposit.임대건물구분[i]) & (data.공급유형==na_security_deposit.공급유형[i]) & (data.자격유형==na_security_deposit.자격유형[i]) & (data.지역==na_security_deposit.지역[i]) & (data.전용면적>=na_security_deposit.전용면적[i]-5) & (data.전용면적<=na_security_deposit.전용면적[i]+5)].임대보증금.mean()
```


```python
data.loc[(data.자격유형!='D')&(data.임대료.isna())].단지코드.unique()
```




    array(['C1326', 'C1786', 'C2186'], dtype=object)




```python
data=data.loc[(data.단지코드!='C1326') & (data.단지코드!='C1786') & (data.단지코드!='C2186')]
```


```python
#지하철역 수, 버스정류장 수 결측치는 최빈값으로 대체한다.
data=fillna_(data, ['자격유형','도보 10분거리 내 지하철역 수(환승노선 수 반영)','도보 10분거리 내 버스정류장 수'], 'most_frequent')
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    

## 데이터 변환


```python
df=data[['임대건물구분','공급유형','자격유형']].drop_duplicates().reset_index(drop=True)
```


```python
for i in range(df.shape[0]):
  data[df.임대건물구분[i]+'_'+df.공급유형[i]+"_"+df.자격유형[i]]=0
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


```python
for i in range(df.shape[0]):
  data.loc[(data.임대건물구분==df.임대건물구분[i])&(data.공급유형==df.공급유형[i])&(data.자격유형==df.자격유형[i]),df.임대건물구분[i]+'_'+df.공급유형[i]+"_"+df.자격유형[i]]=data.loc[(data.임대건물구분==df.임대건물구분[i])&(data.공급유형==df.공급유형[i])&(data.자격유형==df.자격유형[i]),'전용면적별세대수']
```

    /usr/local/lib/python3.7/dist-packages/pandas/core/indexing.py:1743: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      isetter(ilocs[0], value)
    


```python
data['총 면적']=data.전용면적*data.전용면적별세대수
```


```python
new_data=pd.DataFrame()
new_data=pd.concat([new_data,data.groupby('단지코드')['총 면적'].sum()],axis=1)
new_data=pd.concat([new_data,data.groupby('단지코드')['임대보증금'].sum()],axis=1)
new_data=pd.concat([new_data,data.groupby('단지코드')['임대료'].sum()],axis=1)
for i in range(df.shape[0]):
  new_data=pd.concat([new_data,data.groupby('단지코드')[df.임대건물구분[i]+'_'+df.공급유형[i]+"_"+df.자격유형[i]].sum()],axis=1)
```


```python
data_1=data[['단지코드','총세대수','지역','공가수','도보 10분거리 내 지하철역 수(환승노선 수 반영)','도보 10분거리 내 버스정류장 수','단지내주차면수','등록차량수']].drop_duplicates('단지코드')
data_1.index=data_1.단지코드
data_1.drop('단지코드',axis=1,inplace=True)
final_data=pd.concat([data_1,new_data],axis=1)
```


```python
final_data
```




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
      <th>총세대수</th>
      <th>지역</th>
      <th>공가수</th>
      <th>도보 10분거리 내 지하철역 수(환승노선 수 반영)</th>
      <th>도보 10분거리 내 버스정류장 수</th>
      <th>단지내주차면수</th>
      <th>등록차량수</th>
      <th>총 면적</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>아파트_국민임대_A</th>
      <th>아파트_국민임대_BGH</th>
      <th>아파트_공공임대_장기전세_A</th>
      <th>아파트_영구임대_CFI</th>
      <th>상가_임대상가_공공분양_D</th>
      <th>아파트_영구임대_E</th>
      <th>아파트_국민임대_E</th>
      <th>아파트_임대상가_공공분양_D</th>
      <th>아파트_영구임대_A</th>
      <th>아파트_행복주택_JKLMNO</th>
      <th>아파트_영구임대_D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C2515</th>
      <td>545</td>
      <td>2</td>
      <td>17.0</td>
      <td>0</td>
      <td>3</td>
      <td>624.0</td>
      <td>205.0</td>
      <td>21941.82</td>
      <td>1.175100e+08</td>
      <td>9.769000e+05</td>
      <td>545</td>
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
      <th>C1407</th>
      <td>1216</td>
      <td>6</td>
      <td>13.0</td>
      <td>1</td>
      <td>1</td>
      <td>1285.0</td>
      <td>1064.0</td>
      <td>49159.08</td>
      <td>3.136160e+08</td>
      <td>1.944030e+06</td>
      <td>1216</td>
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
      <th>C1945</th>
      <td>755</td>
      <td>1</td>
      <td>6.0</td>
      <td>1</td>
      <td>3</td>
      <td>734.0</td>
      <td>730.0</td>
      <td>37962.15</td>
      <td>1.359100e+08</td>
      <td>1.134640e+06</td>
      <td>0</td>
      <td>755</td>
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
      <th>C1470</th>
      <td>696</td>
      <td>12</td>
      <td>14.0</td>
      <td>0</td>
      <td>2</td>
      <td>645.0</td>
      <td>553.0</td>
      <td>31250.88</td>
      <td>7.089100e+07</td>
      <td>4.806000e+05</td>
      <td>696</td>
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
      <th>C1898</th>
      <td>566</td>
      <td>12</td>
      <td>9.0</td>
      <td>0</td>
      <td>5</td>
      <td>517.0</td>
      <td>415.0</td>
      <td>24174.04</td>
      <td>1.045340e+08</td>
      <td>7.148900e+05</td>
      <td>566</td>
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
    </tr>
    <tr>
      <th>C2456</th>
      <td>346</td>
      <td>13</td>
      <td>17.0</td>
      <td>0</td>
      <td>4</td>
      <td>270.0</td>
      <td>NaN</td>
      <td>12654.70</td>
      <td>4.891500e+07</td>
      <td>6.203300e+05</td>
      <td>0</td>
      <td>346</td>
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
      <th>C1266</th>
      <td>591</td>
      <td>15</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>593.0</td>
      <td>NaN</td>
      <td>21191.31</td>
      <td>7.177600e+07</td>
      <td>9.993900e+05</td>
      <td>0</td>
      <td>591</td>
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
      <th>C2152</th>
      <td>120</td>
      <td>0</td>
      <td>9.0</td>
      <td>0</td>
      <td>1</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>3466.14</td>
      <td>1.230611e+07</td>
      <td>1.811275e+05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C1267</th>
      <td>670</td>
      <td>2</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
      <td>467.0</td>
      <td>NaN</td>
      <td>21636.46</td>
      <td>2.525879e+08</td>
      <td>1.209023e+06</td>
      <td>0</td>
      <td>310</td>
      <td>0</td>
      <td>110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>250</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C2189</th>
      <td>378</td>
      <td>12</td>
      <td>45.0</td>
      <td>0</td>
      <td>2</td>
      <td>300.0</td>
      <td>NaN</td>
      <td>14128.74</td>
      <td>4.584300e+07</td>
      <td>5.239000e+05</td>
      <td>0</td>
      <td>378</td>
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
  </tbody>
</table>
<p>560 rows × 21 columns</p>
</div>



## final_data 탐색


```python
final_data.describe()
```




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
      <th>총세대수</th>
      <th>지역</th>
      <th>공가수</th>
      <th>도보 10분거리 내 지하철역 수(환승노선 수 반영)</th>
      <th>도보 10분거리 내 버스정류장 수</th>
      <th>단지내주차면수</th>
      <th>등록차량수</th>
      <th>총 면적</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>아파트_국민임대_A</th>
      <th>아파트_국민임대_BGH</th>
      <th>아파트_공공임대_장기전세_A</th>
      <th>아파트_영구임대_CFI</th>
      <th>상가_임대상가_공공분양_D</th>
      <th>아파트_영구임대_E</th>
      <th>아파트_국민임대_E</th>
      <th>아파트_임대상가_공공분양_D</th>
      <th>아파트_영구임대_A</th>
      <th>아파트_행복주택_JKLMNO</th>
      <th>아파트_영구임대_D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>410.000000</td>
      <td>560.000000</td>
      <td>5.600000e+02</td>
      <td>5.600000e+02</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
      <td>560.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>707.983929</td>
      <td>5.633929</td>
      <td>13.844643</td>
      <td>0.142857</td>
      <td>3.155357</td>
      <td>575.557143</td>
      <td>566.307317</td>
      <td>30586.554643</td>
      <td>1.437737e+08</td>
      <td>1.051844e+06</td>
      <td>431.273214</td>
      <td>67.276786</td>
      <td>66.819643</td>
      <td>80.462500</td>
      <td>1.319643</td>
      <td>0.967857</td>
      <td>11.580357</td>
      <td>1.269643</td>
      <td>1.551786</td>
      <td>44.091071</td>
      <td>0.046429</td>
    </tr>
    <tr>
      <th>std</th>
      <td>403.024302</td>
      <td>5.069103</td>
      <td>10.465847</td>
      <td>0.350240</td>
      <td>1.359709</td>
      <td>348.498482</td>
      <td>388.880266</td>
      <td>17308.720255</td>
      <td>1.314817e+08</td>
      <td>8.998099e+05</td>
      <td>397.972660</td>
      <td>250.477234</td>
      <td>215.388490</td>
      <td>302.842813</td>
      <td>5.532435</td>
      <td>19.219820</td>
      <td>74.908321</td>
      <td>30.045234</td>
      <td>31.905846</td>
      <td>197.574290</td>
      <td>1.098701</td>
    </tr>
    <tr>
      <th>min</th>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>711.140000</td>
      <td>3.056000e+06</td>
      <td>6.800000e+04</td>
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
    </tr>
    <tr>
      <th>25%</th>
      <td>420.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>316.750000</td>
      <td>285.000000</td>
      <td>18207.777500</td>
      <td>6.223275e+07</td>
      <td>5.217675e+05</td>
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
    </tr>
    <tr>
      <th>50%</th>
      <td>619.500000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>512.000000</td>
      <td>505.000000</td>
      <td>27271.485000</td>
      <td>1.040485e+08</td>
      <td>7.707150e+05</td>
      <td>406.500000</td>
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
    </tr>
    <tr>
      <th>75%</th>
      <td>919.750000</td>
      <td>11.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>773.500000</td>
      <td>756.250000</td>
      <td>39581.070000</td>
      <td>1.845315e+08</td>
      <td>1.312485e+06</td>
      <td>690.750000</td>
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
    </tr>
    <tr>
      <th>max</th>
      <td>2572.000000</td>
      <td>15.000000</td>
      <td>55.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1798.000000</td>
      <td>2550.000000</td>
      <td>94979.630000</td>
      <td>1.037727e+09</td>
      <td>8.560000e+06</td>
      <td>1722.000000</td>
      <td>2334.000000</td>
      <td>1444.000000</td>
      <td>2529.000000</td>
      <td>45.000000</td>
      <td>450.000000</td>
      <td>798.000000</td>
      <td>711.000000</td>
      <td>745.000000</td>
      <td>2200.000000</td>
      <td>26.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 공가율 파생변수 생성
final_data['공가율']=final_data.공가수/final_data.총세대수
```


```python
discrete_variables=['지역','도보 10분거리 내 지하철역 수(환승노선 수 반영)',
       '도보 10분거리 내 버스정류장 수']
continuous_variable=['등록차량수','총세대수', '공가수', '단지내주차면수','총 면적', '임대보증금', '임대료', '공가율']
```


```python
sns.pairplot(final_data.loc[:,continuous_variable])
plt.title("Continuous Variable")
plt.show()
```


    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_64_0.png' height = '400' width = '600'>
    



```python
#TrainSet과 Test으로 구분
new_train=pd.DataFrame()
new_test=pd.DataFrame()

for code in list(train.단지코드.unique()):
  new_train=new_train.append(final_data.loc[final_data.index==code])

for code in list(test.단지코드.unique()):
  new_test=new_test.append(final_data.loc[final_data.index==code])
```


```python
sns.distplot(new_train['등록차량수'])
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5b341a4890>




    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_66_2.png' height = '400' width = '600'>
    


# 3. 모델링


```python
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
```


```python
s_scaler=StandardScaler()
train_y=new_train['등록차량수']
train_X=new_train.drop('등록차량수',axis=1)
test_X=new_test.drop('등록차량수',axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.1, random_state=12)
```

회귀모델


```python
def cv_rmse(model,x,y):
    rmse = np.sqrt(-cross_val_score(model, s_scaler.fit_transform(x) , y, 
                                   scoring="neg_mean_squared_error", 
                                   cv = 5))
    return(rmse)

def ridge_selector(k,x,y):
    ridge_model = make_pipeline(RidgeCV(alphas = [k],
                                        cv=5)).fit(s_scaler.fit_transform(x), y)
    ridge_rmse = cv_rmse(ridge_model,x,y).mean()
    return(ridge_rmse)
```


```python
r_alphas = [4.8,4.85,4.9,4.95,5,5.05,5.1]

ridge_scores = []
for alpha in r_alphas:
    score = ridge_selector(alpha, X_train, y_train)
    ridge_scores.append(score)
    
plt.plot(r_alphas, ridge_scores, label='Ridge')
plt.legend('center')
plt.xlabel('alpha')
plt.ylabel('score')

ridge_score_table = pd.DataFrame(ridge_scores, r_alphas, columns=['RMSE'])
ridge_score_table
```



  <div id="df-355c5119-c3e2-4538-bd96-baf381dde928">

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
<table border="0" class="">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4.80</th>
      <td>182.604539</td>
    </tr>
    <tr>
      <th>4.85</th>
      <td>182.604194</td>
    </tr>
    <tr>
      <th>4.90</th>
      <td>182.603982</td>
    </tr>
    <tr>
      <th>4.95</th>
      <td>182.603900</td>
    </tr>
    <tr>
      <th>5.00</th>
      <td>182.603944</td>
    </tr>
    <tr>
      <th>5.05</th>
      <td>182.604112</td>
    </tr>
    <tr>
      <th>5.10</th>
      <td>182.604400</td>
    </tr>
  </tbody>
</table>
</div>
</div>



    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_72_1.png' height = '400' width = '600'>
    



```python
RIDGE=Ridge(alpha=4.95,random_state=12)
RIDGE.fit(s_scaler.fit_transform(X_train),y_train)
pred_RIDGE=RIDGE.predict(s_scaler.fit_transform(X_test))
```

부스팅,  붓스트랩 트리


```python
from sklearn.linear_model import  Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
# 데이터를 로딩하고 학습데이타와 테스트 데이터 분리

XGB = XGBRegressor()
RF = RandomForestRegressor()

XGB_param = {
    'learning_rate':[0.1,0.2], 
    'n_estimators':[100,150,200],
    'max_depth':[2,4,6,8], 
    'subsample':[0.5,0.7,0.9], 
    'colsample_bytree':[0.5,0.7,0.9],
    'min_samples_split':[2,3]
    }
RF_param = {
    'n_estimators':[100,150,200],
    'max_depth':[2,4,6,8], 
    'min_samples_leaf':[2,3],
    'min_samples_split':[2,3,4]
    }
```


```python
### refit=True 가 default 임. True이면 가장 좋은 파라미터 설정으로 재 학습 시킴.  
grid_XGB = GridSearchCV(XGB, param_grid=XGB_param, cv=5, refit=True, return_train_score=True,
                        scoring='neg_mean_squared_error')
grid_XGB.fit(X_train, y_train)

grid_RF = GridSearchCV(RF, param_grid=RF_param, cv=5, refit=True, return_train_score=True,
                       scoring='neg_mean_squared_error')
grid_RF.fit(X_train, y_train)

# GridSearchCV 결과는 cv_results_ 라는 딕셔너리로 저장됨. 이를 DataFrame으로 변환
scores_df_XGB = pd.DataFrame(grid_XGB.cv_results_)
scores_df_RF = pd.DataFrame(grid_RF.cv_results_)
```


```python
scores_df_XGB.mean_test_score=np.sqrt(-scores_df_XGB.mean_test_score)
scores_df_XGB.std_train_score=np.sqrt(scores_df_XGB.std_train_score)
scores_df_XGB.split0_test_score=np.sqrt(-scores_df_XGB.split0_test_score)
scores_df_XGB.split1_test_score=np.sqrt(-scores_df_XGB.split1_test_score)
scores_df_XGB.split2_test_score=np.sqrt(-scores_df_XGB.split2_test_score)
scores_df_XGB.split3_test_score=np.sqrt(-scores_df_XGB.split3_test_score)
scores_df_XGB.split4_test_score=np.sqrt(-scores_df_XGB.split4_test_score)

scores_df_RF.mean_test_score=np.sqrt(-scores_df_RF.mean_test_score)
scores_df_RF.std_train_score=np.sqrt(scores_df_RF.std_train_score)
scores_df_RF.split0_test_score=np.sqrt(-scores_df_RF.split0_test_score)
scores_df_RF.split1_test_score=np.sqrt(-scores_df_RF.split1_test_score)
scores_df_RF.split2_test_score=np.sqrt(-scores_df_RF.split2_test_score)
scores_df_RF.split3_test_score=np.sqrt(-scores_df_RF.split3_test_score)
scores_df_RF.split4_test_score=np.sqrt(-scores_df_RF.split4_test_score)
```


```python
scores_df_XGB[['rank_test_score','params', 'mean_test_score','std_train_score',  
           'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']].sort_values('rank_test_score').head(5)
```




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
      <th>rank_test_score</th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_train_score</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>289</th>
      <td>1</td>
      <td>{'colsample_bytree': 0.9, 'learning_rate': 0.1...</td>
      <td>194.444645</td>
      <td>15.427918</td>
      <td>195.077831</td>
      <td>194.651648</td>
      <td>169.835584</td>
      <td>207.111684</td>
      <td>203.370600</td>
    </tr>
    <tr>
      <th>298</th>
      <td>1</td>
      <td>{'colsample_bytree': 0.9, 'learning_rate': 0.1...</td>
      <td>194.444645</td>
      <td>15.427918</td>
      <td>195.077831</td>
      <td>194.651648</td>
      <td>169.835584</td>
      <td>207.111684</td>
      <td>203.370600</td>
    </tr>
    <tr>
      <th>304</th>
      <td>3</td>
      <td>{'colsample_bytree': 0.9, 'learning_rate': 0.1...</td>
      <td>195.339740</td>
      <td>18.930358</td>
      <td>198.207641</td>
      <td>196.705159</td>
      <td>167.280758</td>
      <td>207.131032</td>
      <td>204.750495</td>
    </tr>
    <tr>
      <th>295</th>
      <td>3</td>
      <td>{'colsample_bytree': 0.9, 'learning_rate': 0.1...</td>
      <td>195.339740</td>
      <td>18.930358</td>
      <td>198.207641</td>
      <td>196.705159</td>
      <td>167.280758</td>
      <td>207.131032</td>
      <td>204.750495</td>
    </tr>
    <tr>
      <th>292</th>
      <td>5</td>
      <td>{'colsample_bytree': 0.9, 'learning_rate': 0.1...</td>
      <td>195.891849</td>
      <td>19.189187</td>
      <td>198.674076</td>
      <td>194.975916</td>
      <td>167.632427</td>
      <td>207.965433</td>
      <td>207.438746</td>
    </tr>
  </tbody>
</table>
</div>




```python
scores_df_RF[['rank_test_score','params', 'mean_test_score','std_train_score',  
           'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']].sort_values('rank_test_score').head(5)
```




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
      <th>rank_test_score</th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_train_score</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>1</td>
      <td>{'max_depth': 8, 'min_samples_leaf': 2, 'min_s...</td>
      <td>197.660758</td>
      <td>24.067901</td>
      <td>184.271782</td>
      <td>216.183980</td>
      <td>178.044176</td>
      <td>214.225915</td>
      <td>192.522212</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2</td>
      <td>{'max_depth': 6, 'min_samples_leaf': 2, 'min_s...</td>
      <td>198.266168</td>
      <td>20.798448</td>
      <td>188.215324</td>
      <td>211.943622</td>
      <td>181.268349</td>
      <td>215.667008</td>
      <td>191.916092</td>
    </tr>
    <tr>
      <th>38</th>
      <td>3</td>
      <td>{'max_depth': 6, 'min_samples_leaf': 2, 'min_s...</td>
      <td>198.952667</td>
      <td>19.316410</td>
      <td>183.559045</td>
      <td>218.173595</td>
      <td>182.444971</td>
      <td>214.840380</td>
      <td>192.807216</td>
    </tr>
    <tr>
      <th>55</th>
      <td>4</td>
      <td>{'max_depth': 8, 'min_samples_leaf': 2, 'min_s...</td>
      <td>199.117951</td>
      <td>24.546255</td>
      <td>185.451063</td>
      <td>217.226329</td>
      <td>177.956940</td>
      <td>217.898109</td>
      <td>193.680559</td>
    </tr>
    <tr>
      <th>60</th>
      <td>5</td>
      <td>{'max_depth': 8, 'min_samples_leaf': 2, 'min_s...</td>
      <td>199.172112</td>
      <td>22.694906</td>
      <td>182.327208</td>
      <td>215.607289</td>
      <td>188.745992</td>
      <td>211.737916</td>
      <td>195.345700</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('XGB GridSearchCV 최적 파라미터:', grid_XGB.best_params_)
print('XGB GridSearchCV 최고 정확도: {0:.4f}'.format(np.sqrt(-grid_XGB.best_score_)))

print('RF GridSearchCV 최적 파라미터:', grid_RF.best_params_)
print('RF GridSearchCV 최고 정확도: {0:.4f}'.format(np.sqrt(-grid_RF.best_score_)))


# refit=True로 설정된 GridSearchCV 객체가 fit()을 수행 시 학습이 완료된 Estimator를 내포하고 있으므로 predict()를 통해 예측도 가능. 
pred_XGB = grid_XGB.predict(X_test)
pred_RF = grid_RF.predict(X_test)

print('XGB : 테스트 데이터 세트 정확도: {0:.4f}'.format(np.sqrt(mean_squared_error(y_test,pred_XGB))))
print('RF : 테스트 데이터 세트 정확도: {0:.4f}'.format(np.sqrt(mean_squared_error(y_test,pred_RF))))
print('RIDGE : 테스트 데이터 세트 정확도: {0:.4f}'.format(np.sqrt(mean_squared_error(y_test,pred_RIDGE))))
```

    XGB GridSearchCV 최적 파라미터: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 2, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.7}
    XGB GridSearchCV 최고 정확도: 194.4446
    RF GridSearchCV 최적 파라미터: {'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    RF GridSearchCV 최고 정확도: 197.6608
    XGB : 테스트 데이터 세트 정확도: 152.3748
    RF : 테스트 데이터 세트 정확도: 149.9082
    RIDGE : 테스트 데이터 세트 정확도: 154.9575
    


```python
votingC = VotingRegressor(estimators=[('XGB',grid_XGB.best_estimator_),('RF',grid_RF.best_estimator_) ,('Ridge', RIDGE)], n_jobs=-1)
votingC = votingC.fit(s_scaler.fit_transform(X_train),y_train)
pred_voting=votingC.predict(s_scaler.fit_transform(X_test))
```


```python
plot_importance(grid_XGB.best_estimator_)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5b2ba5a090>




    
<img src = '/images/2021-12-31-AI_Competition_to_Predict_Parking_Demand_files/2021-12-31-AI_Competition_to_Predict_Parking_Demand_82_1.png' height = '400' width = '600'>
    



```python
print('Voting : 테스트 데이터 세트 정확도: {0:.4f}'.format(np.sqrt(mean_squared_error(y_test,pred_voting))))
```

    Voting : 테스트 데이터 세트 정확도: 150.2447
    
