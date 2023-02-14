---
title: "Heart Attack Analysis & Prediction Dataset"
tags: [Data Anaylsis, Code Review, Machine Learning]
use_math: true
comments: true
date : 2021-09-01
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
Kaggle의 Heart Attack Analysis & Prediction Dataset대회이다. 간단한 EDA 후 바로 모델링을 하였다. **make_pipeline**을 통해 스케쥴러, **SVC**모델을 적합시켰다. 또한 **GridSearchCV**를 통해 **linear kernel**, **rbf** 커널의 SVC모델을 적합시켰다.

# 1.Library & Data Load

```python
from google.colab import drive
drive.mount('/content/drive') 
```

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')
```


```python
heart=pd.read_csv('/content/drive/MyDrive/Kaggle/Data/Heart Attack Analysis & Prediction Dataset/heart.csv')
o2Saturation=pd.read_csv('/content/drive/MyDrive/Kaggle/Data/Heart Attack Analysis & Prediction Dataset/o2Saturation.csv')
```


```python
heart.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       303 non-null    int64  
     1   sex       303 non-null    int64  
     2   cp        303 non-null    int64  
     3   trtbps    303 non-null    int64  
     4   chol      303 non-null    int64  
     5   fbs       303 non-null    int64  
     6   restecg   303 non-null    int64  
     7   thalachh  303 non-null    int64  
     8   exng      303 non-null    int64  
     9   oldpeak   303 non-null    float64
     10  slp       303 non-null    int64  
     11  caa       303 non-null    int64  
     12  thall     303 non-null    int64  
     13  output    303 non-null    int64  
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB
    


```python
heart.drop_duplicates(inplace=True)
```


```python
heart.describe()
```





  <div id="df-5841e5fa-c660-43e5-bc86-79f7d1f58500">
    <div class = ".colab-df-container">
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>302.00000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.42053</td>
      <td>0.682119</td>
      <td>0.963576</td>
      <td>131.602649</td>
      <td>246.500000</td>
      <td>0.149007</td>
      <td>0.526490</td>
      <td>149.569536</td>
      <td>0.327815</td>
      <td>1.043046</td>
      <td>1.397351</td>
      <td>0.718543</td>
      <td>2.314570</td>
      <td>0.543046</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.04797</td>
      <td>0.466426</td>
      <td>1.032044</td>
      <td>17.563394</td>
      <td>51.753489</td>
      <td>0.356686</td>
      <td>0.526027</td>
      <td>22.903527</td>
      <td>0.470196</td>
      <td>1.161452</td>
      <td>0.616274</td>
      <td>1.006748</td>
      <td>0.613026</td>
      <td>0.498970</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>211.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>133.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.50000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>130.000000</td>
      <td>240.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>152.500000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.00000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>140.000000</td>
      <td>274.750000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>166.000000</td>
      <td>1.000000</td>
      <td>1.600000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.00000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>200.000000</td>
      <td>564.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>202.000000</td>
      <td>1.000000</td>
      <td>6.200000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5841e5fa-c660-43e5-bc86-79f7d1f58500')"
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
          document.querySelector('#df-5841e5fa-c660-43e5-bc86-79f7d1f58500 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5841e5fa-c660-43e5-bc86-79f7d1f58500');
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
plt.figure(figsize=(16,8))
sns.heatmap(heart.corr(),annot=True)
plt.show()
```


    
![png](/images/2021-12-31-Heart_Attack_Analysis_Prediction_Dataset_files/2021-12-31-Heart_Attack_Analysis_Prediction_Dataset_9_0.png)
    

<br><br>

# 2. 모델링


```python
X = heart.iloc[:, 0: -1]
y = heart.iloc[:, -1:]
```


```python
X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=2)
```


```python
pipe_svc=make_pipeline(StandardScaler(),
                       SVC(random_state=1))
param_C_range=[0,1,2,3,4,5,6,7,8,9,10]
param_gamma_range=[0.005,0, 0.1, 0.015]

param_grid=[{'svc__C' : param_C_range,
             'svc__kernel':['linear']},
            {'svc__C':param_C_range,
             'svc__gamma':param_gamma_range,
             'svc__kernel':['rbf']                
            }]
gs=GridSearchCV(estimator=pipe_svc,
                param_grid=param_grid,
                scoring='accuracy',
                cv=10,
                n_jobs=-1)

gs=gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)
```

    0.8463333333333335
    {'svc__C': 4, 'svc__gamma': 0.015, 'svc__kernel': 'rbf'}
    


```python
clf=gs.best_estimator_
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test) )
```

    0.8524590163934426
    


```python
confusion_matrix(y_test,clf.predict(X_test))
```




    array([[25,  8],
           [ 1, 27]])




```python
clf.score(X_test,y_test)
```




    0.8524590163934426


