---
title: "A contest to predict the rental volume of Ttareungi"
tags: [Data Anaylsis, Machine Learning]
comments: true
date : 2021-12-31
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

    Mounted at /content/drive
    


```python
# !mkdir "/content/drive/MyDrive/DACON/Data/A contest to predict the rental volume of Ttareungi"
# !unzip "/content/drive/MyDrive/DACON/Data/A contest to predict the rental volume of Ttareungi.zip" -d "/content/drive/MyDrive/DACON/Data/A contest to predict the rental volume of Ttareungi"
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
%matplotlib inline 
```


```python
path='/content/drive/MyDrive/DACON/Data/A contest to predict the rental volume of Ttareungi/'
```


```python
train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
submission=pd.read_csv(path+'sample_submission.csv')
```

# 2. Data Preprocess


```python
train.date_time=pd.to_datetime(train.date_time)
train['week'] = pd.to_datetime(train['date_time']).dt.week
train['weekday'] = pd.to_datetime(train['date_time']).dt.weekday

test.date_time=pd.to_datetime(test.date_time)
test['week'] = pd.to_datetime(test['date_time']).dt.week
test['weekday'] = pd.to_datetime(test['date_time']).dt.weekday
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.
      
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.
      
    


```python
train['date']=0
train.loc[train.date_time.dt.year==2018,'date']=range(0,91)
train.loc[train.date_time.dt.year==2019,'date']=range(0,91)
train.loc[train.date_time.dt.year==2020,'date']=range(0,91)
test['date']=range(0,91)
```


```python
def draw_plot(data):
  count=1
  for year in [2018,2019,2020]:
    plt.subplot(3,1,count)
    plt.plot(data.loc[(data.date_time>=datetime(year,1,1)) & (data.date_time<=datetime(year,12,31)),'number_of_rentals'])
    count=count+1
    plt.show()
```


```python
def min_max(data):
  data=data.reset_index(drop=True)
  new_data=[]
  for i in range(data.shape[0]):
    new_data.append((data[i]-data.min())/(data.max()-data.min()))
  return new_data
```


```python
def return_min_max(data,_max,_min):
  data=data.reset_index(drop=True)
  new_data=[]
  for i in range(data.shape[0]):
    new_data.append(data[i]*(_max-_min)+_min)
  return new_data
```


```python
for year in [2018,2019,2020]:
  print(train.loc[train.date_time.dt.year==year,'number_of_rentals'].max())
for year in [2018,2019,2020]:
  print(train.loc[train.date_time.dt.year==year,'number_of_rentals'].min())
```

    49116
    88432
    110377
    1037
    12749
    7600
    


```python
train.loc[(train.date_time.dt.year==2020)&(train.number_of_rentals<20000)]
```





  <div id="df-40a8bc8a-975f-4cd5-9051-ab1d51383537">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
      <th>week</th>
      <th>weekday</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>220</th>
      <td>2020-05-09</td>
      <td>144.142</td>
      <td>4.000</td>
      <td>1.000</td>
      <td>4.192</td>
      <td>80.034</td>
      <td>13.938</td>
      <td>21.158</td>
      <td>82.162</td>
      <td>7600</td>
      <td>19</td>
      <td>5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>226</th>
      <td>2020-05-15</td>
      <td>140.966</td>
      <td>3.865</td>
      <td>0.723</td>
      <td>3.804</td>
      <td>69.122</td>
      <td>15.875</td>
      <td>22.263</td>
      <td>61.216</td>
      <td>13782</td>
      <td>20</td>
      <td>4</td>
      <td>44</td>
    </tr>
    <tr>
      <th>266</th>
      <td>2020-06-24</td>
      <td>124.797</td>
      <td>3.973</td>
      <td>0.716</td>
      <td>3.203</td>
      <td>76.182</td>
      <td>21.375</td>
      <td>26.421</td>
      <td>62.500</td>
      <td>19756</td>
      <td>26</td>
      <td>2</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-40a8bc8a-975f-4cd5-9051-ab1d51383537')"
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
          document.querySelector('#df-40a8bc8a-975f-4cd5-9051-ab1d51383537 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-40a8bc8a-975f-4cd5-9051-ab1d51383537');
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
draw_plot(train)
```


    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_15_0.png' height = '400' width = '600'>
    



    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_15_1.png' height = '400' width = '600'>
    



    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_15_2.png' height = '400' width = '600'>
    



```python
train.loc[train.date_time.dt.year==2018,'number_of_rentals']=min_max(train.loc[train.date_time.dt.year==2018,'number_of_rentals'])
train.loc[train.date_time.dt.year==2019,'number_of_rentals']=min_max(train.loc[train.date_time.dt.year==2019,'number_of_rentals'])
train.loc[train.date_time.dt.year==2020,'number_of_rentals']=min_max(train.loc[train.date_time.dt.year==2020,'number_of_rentals'])
```


```python
new_data=train.copy()
```


```python
draw_plot(train)
```


    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_18_0.png' height = '400' width = '600'>
    



    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_18_1.png' height = '400' width = '600'>
    



    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_18_2.png' height = '400' width = '600'>
    



```python
train.loc[(train.date_time>=datetime(2018,1,1)) & (train.date_time<=datetime(2018,12,31)),'number_of_rentals']=return_min_max(train.loc[(train.date_time>=datetime(2018,1,1)) & (train.date_time<=datetime(2018,12,31)),'number_of_rentals'],49116,1037)
train.loc[(train.date_time>=datetime(2019,1,1)) & (train.date_time<=datetime(2019,12,31)),'number_of_rentals']=return_min_max(train.loc[(train.date_time>=datetime(2019,1,1)) & (train.date_time<=datetime(2019,12,31)),'number_of_rentals'],88432,1037)
train.loc[(train.date_time>=datetime(2020,1,1)) & (train.date_time<=datetime(2020,12,31)),'number_of_rentals']=return_min_max(train.loc[(train.date_time>=datetime(2020,1,1)) & (train.date_time<=datetime(2020,12,31)),'number_of_rentals'],110377,1037)
```


```python
draw_plot(train)
```


    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_20_0.png' height = '400' width = '600'>
    



    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_20_1.png' height = '400' width = '600'>
    



    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_20_2.png' height = '400' width = '600'>
    



```python
train_label=new_data['number_of_rentals']
train_X=new_data.drop(['wind_direction','precipitation_form','date_time','number_of_rentals','date','week'],axis=1)
test.drop(['wind_direction','precipitation_form','date_time','date','week'],axis=1,inplace=True)
```

# 3. 모델링



```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 0, n_estimators = 100)
rf.fit(train_X,train_label)
predict=rf.predict(test)
```


```python
_max=150000
_min=1037
data=predict
new_data=[]
for i in range(data.shape[0]):
  new_data.append(data[i]*(_max-_min)+_min)
plt.plot(new_data)
```




    [<matplotlib.lines.Line2D at 0x7f8c5a43b910>]




    
<img src = '/images/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_files/2021-12-31-A_contest_to_predict_the_rental_volume_of_Ttareungi_24_1.png' height = '400' width = '600'>
    



```python
submission['number_of_rentals']=new_data
submission.to_csv(path+'submission17.csv',index=False)
```
