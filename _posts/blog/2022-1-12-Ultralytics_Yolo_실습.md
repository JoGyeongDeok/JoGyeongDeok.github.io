---
title: "Ultralytics Yolo 실습"
category: DeepLearning
tags: [Computer Vision, Object Detection, Deep Learning, Yolo]
comments: true
date : 2022-01-12
categories: 
  - blog
excerpt: CV
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

<br>

# 1. Ultralytics Yolo v3 & Library & Data 설치


```python
!git clone https://github.com/ultralytics/yolov3
!cd yolov3; pip install -qr requirements.txt
```

    fatal: destination path 'yolov3' already exists and is not an empty directory.
    


```python
import torch
from IPython.display import Image, clear_output # to display images

clear_output()
print(f"Set up complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

    Set up complete. Using torch 1.10.0+cu111 (CPU)
    

OXford Pet Dataset 다운로드


```python
# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

    --2022-01-12 08:24:31--  https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    Resolving www.robots.ox.ac.uk (www.robots.ox.ac.uk)... 129.67.94.2
    Connecting to www.robots.ox.ac.uk (www.robots.ox.ac.uk)|129.67.94.2|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 791918971 (755M) [application/x-gzip]
    Saving to: ‘images.tar.gz’
    
    images.tar.gz       100%[===================>] 755.23M  29.4MB/s    in 28s     
    
    2022-01-12 08:24:59 (27.4 MB/s) - ‘images.tar.gz’ saved [791918971/791918971]
    
    --2022-01-12 08:24:59--  https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    Resolving www.robots.ox.ac.uk (www.robots.ox.ac.uk)... 129.67.94.2
    Connecting to www.robots.ox.ac.uk (www.robots.ox.ac.uk)|129.67.94.2|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19173078 (18M) [application/x-gzip]
    Saving to: ‘annotations.tar.gz’
    
    annotations.tar.gz  100%[===================>]  18.28M  12.5MB/s    in 1.5s    
    
    2022-01-12 08:25:01 (12.5 MB/s) - ‘annotations.tar.gz’ saved [19173078/19173078]
    
    


```python
# # /content/data 디렉토리를 만들고 해당 디렉토리에 다운로드 받은 압축 파일 풀기.
# !mkdir '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet'
# !mkdir '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data'
# !tar -xvf images.tar.gz -C '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data'
# !tar -xvf annotations.tar.gz -C '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data'
```

Oxford Pet Dataset 리뷰 및 학습/검증 데이터 세트로 분리


```python
# Ultralytics Yolo images와 labels 디렉토리를 train, val 용으로 생성
# !mkdir '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet'
# !cd '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet'; mkdir images; mkdir labels;
# !cd '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/images'; mkdir train; mkdir val
# !cd '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/labels'; mkdir train; mkdir val
```


```python
import pandas as pd

pd.read_csv('/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data/annotations/trainval.txt', sep=' ', header=None, names=['img_name', 'class_id', 'etc1', 'etc2'])
```





  <div id="df-36eec691-8b52-473e-826b-665210ff8948">
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
      <th>img_name</th>
      <th>class_id</th>
      <th>etc1</th>
      <th>etc2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abyssinian_100</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abyssinian_101</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abyssinian_102</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abyssinian_103</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abyssinian_104</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3675</th>
      <td>yorkshire_terrier_187</td>
      <td>37</td>
      <td>2</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3676</th>
      <td>yorkshire_terrier_188</td>
      <td>37</td>
      <td>2</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3677</th>
      <td>yorkshire_terrier_189</td>
      <td>37</td>
      <td>2</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3678</th>
      <td>yorkshire_terrier_18</td>
      <td>37</td>
      <td>2</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3679</th>
      <td>yorkshire_terrier_190</td>
      <td>37</td>
      <td>2</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
<p>3680 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-36eec691-8b52-473e-826b-665210ff8948')"
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
          document.querySelector('#df-36eec691-8b52-473e-826b-665210ff8948 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-36eec691-8b52-473e-826b-665210ff8948');
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
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 전체 image/annotation 파일명을 가지는 리스트 파일명을 입력 받아 메타 파일용 DataFrame 및 학습/검증용 DataFrame 생성. 
def make_train_valid_df(list_filepath, img_dir, anno_dir, test_size=0.1):
  pet_df = pd.read_csv(list_filepath, sep=' ', header=None, names=['img_name', 'class_id', 'etc1', 'etc2'])
  #class_name은 image 파일명에서 맨 마지막 '_' 문자열 앞까지에 해당. 
  pet_df['class_name'] = pet_df['img_name'].apply(lambda x:x[:x.rfind('_')])
  
  # image 파일명과 annotation 파일명의 절대경로 컬럼 추가
  pet_df['img_filepath'] = img_dir + pet_df['img_name']+'.jpg'
  pet_df['anno_filepath'] = anno_dir + pet_df['img_name']+'.xml'
  # annotation xml 파일이 없는데, trainval.txt에는 리스트가 있는 경우가 있음. 이들의 경우 pet_df에서 해당 rows를 삭제함. 
  pet_df = remove_no_annos(pet_df)

  # 전체 데이터의 10%를 검증 데이터로, 나머지는 학습 데이터로 분리. 
  train_df, val_df = train_test_split(pet_df, test_size=test_size, stratify=pet_df['class_id'], random_state=2021)
  return pet_df, train_df, val_df

# annotation xml 파일이 없는데, trainval.txt에는 리스트가 있는 경우에 이들을 dataframe에서 삭제하기 위한 함수.
def remove_no_annos(df):
  remove_rows = []
  for index, row in df.iterrows():
    anno_filepath = row['anno_filepath']
    if not os.path.exists(anno_filepath):
      print('##### index:', index, anno_filepath, '가 존재하지 않아서 Dataframe에서 삭제함')
      #해당 DataFrame index를 remove_rows list에 담음. 
      remove_rows.append(index)
  # DataFrame의 index가 담긴 list를 drop()인자로 입력하여 해당 rows를 삭제
  df = df.drop(remove_rows, axis=0, inplace=False)
  return df


pet_df, train_df, val_df = make_train_valid_df('/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data/annotations/trainval.txt', 
                                               '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data/images/', '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data/annotations/xmls/', test_size=0.1)
```


```python
pet_df.head()
```





  <div id="df-f992990a-4755-42a7-8f1e-0e21f6535805">
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
      <th>img_name</th>
      <th>class_id</th>
      <th>etc1</th>
      <th>etc2</th>
      <th>class_name</th>
      <th>img_filepath</th>
      <th>anno_filepath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abyssinian_100</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Abyssinian</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abyssinian_101</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Abyssinian</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abyssinian_102</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Abyssinian</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abyssinian_103</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Abyssinian</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Abyssinian_105</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Abyssinian</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
      <td>/content/drive/MyDrive/딥러닝/Computer Visio...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f992990a-4755-42a7-8f1e-0e21f6535805')"
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
          document.querySelector('#df-f992990a-4755-42a7-8f1e-0e21f6535805 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f992990a-4755-42a7-8f1e-0e21f6535805');
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
import os

for index, row in pet_df.iterrows():
  anno_filepath = row['anno_filepath']
  if not os.path.exists(anno_filepath):
    print(anno_filepath)
```
<br>

# 2. Oxford Pet 데이터 세트의 annotation을 Ultralytics Yolo format으로 생성.

- annotation용 xml 파일을 txt 파일로 변환
- 하나의 이미지는 하나의 txt 파일로 변환
- 확장자를 제외한 이미지의 파일명과 annotation 파일명이 서로 동일해야 함.
- 하나의 xml annotation 파일을 Yolo 포맷용 txt 파일로 변환하는 함수 생성
- voc annotation의 좌상단(Top left: x1, y1), 우하단(Bottom right: x2, y2) 좌표를 - Bounding Box 중심 좌표(Center_x, Center_y)와 너비(width), 높이(height)로 변경
- 중심 좌표와 너비, 높이는 원본 이미지 레벨로 scale 되어야 함. 모든 값은 0~1 사이 값으로 변환됨.
- class_id는 여러개의 label들을 0 부터 순차적으로 1씩 증가시켜 id 부여


```python
# Class 명을 부여. Class id는 자동적으로 CLASS_NAMES 개별 원소들을 순차적으로 0부터 36까지 부여
CLASS_NAMES = pet_df['class_name'].unique().tolist()
print(CLASS_NAMES)
```

    ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    


```python
import glob
import xml.etree.ElementTree as ET

# 1개의 voc xml 파일을 Yolo 포맷용 txt 파일로 변경하는 함수 
def xml_to_txt(input_xml_file, output_txt_file, object_name):
  # ElementTree로 입력 XML파일 파싱. 
  tree = ET.parse(input_xml_file)
  root = tree.getroot()
  img_node = root.find('size')
  # img_node를 찾지 못하면 종료
  if img_node is None:
    return None
  # 원본 이미지의 너비와 높이 추출. 
  img_width = int(img_node.find('width').text)
  img_height = int(img_node.find('height').text)

  # xml 파일내에 있는 모든 object Element를 찾음. 
  value_str = None
  with open(output_txt_file, 'w') as output_fpointer:
    for obj in root.findall('object'):
        # bndbox를 찾아서 좌상단(xmin, ymin), 우하단(xmax, ymax) 좌표 추출. 
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        # 만약 좌표중에 하나라도 0보다 작은 값이 있으면 종료. 
        if (x1 < 0) or (x2 < 0) or (y1 < 0) or (y2 < 0):
          break
        # object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환하는 convert_yolo_coord()함수 호출. 
        class_id, cx_norm, cy_norm, w_norm, h_norm = convert_yolo_coord(object_name, img_width, img_height, x1, y1, x2, y2)
        # 변환된 yolo 좌표를 object 별로 출력 text 파일에 write
        value_str = ('{0} {1} {2} {3} {4}').format(class_id-1, cx_norm, cy_norm, w_norm, h_norm)
        output_fpointer.write(value_str+'\n')
        # debugging용으로 아래 출력
        #print(object_name, value_str)

# object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환
def convert_yolo_coord(object_name, img_width, img_height, x1, y1, x2, y2):
  # class_id는 CLASS_NAMES 리스트에서 index 번호로 추출. 
  class_id = CLASS_NAMES.index(object_name)
  # 중심 좌표와 너비, 높이 계산. 
  center_x = (x1 + x2)/2
  center_y = (y1 + y2)/2
  width = x2 - x1
  height = y2 - y1
  # 원본 이미지 기준으로 중심 좌표와 너비 높이를 0-1 사이 값으로 scaling
  center_x_norm = center_x / img_width
  center_y_norm = center_y / img_height
  width_norm = width / img_width
  height_norm = height / img_height

  return class_id, round(center_x_norm, 7), round(center_y_norm, 7), round(width_norm, 7), round(height_norm, 7)
```


```python
class_id = CLASS_NAMES.index('yorkshire_terrier')
print(class_id)
```

    36
    


```python
xml_to_txt('/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/data/annotations/xmls/Abyssinian_1.xml', '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/labels/train/Abyssinian_1.txt', 'Abyssinian')
```

***VOC Format의 여러개 xml 파일들을 Yolo format으로 변환 후 Ultralytics directory 구조로 입력***



- VOC XML 파일들이 있는 디렉토리와 변환하여 출력될 Yolo format txt파일들이 있을 디렉토리를 입력하여 파일들을 생성.



```python
import shutil

def make_yolo_anno_file(df, tgt_images_dir, tgt_labels_dir):
  for index, row in df.iterrows():
    src_image_path = row['img_filepath']
    src_label_path = row['anno_filepath']
    # 이미지 1개당 단 1개의 오브젝트만 존재하므로 class_name을 object_name으로 설정.  
    object_name = row['class_name']
    # yolo format으로 annotation할 txt 파일의 절대 경로명을 지정. 
    target_label_path = tgt_labels_dir + row['img_name']+'.txt'
    # image의 경우 target images 디렉토리로 단순 copy
    shutil.copy(src_image_path, tgt_images_dir)
    # annotation의 경우 xml 파일을 target labels 디렉토리에 Ultralytics Yolo format으로 변환하여  만듬
    xml_to_txt(src_label_path, target_label_path, object_name)

## train용 images와 labels annotation 생성. 
# make_yolo_anno_file(train_df, '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/images/train/', '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/labels/train/')
# # val용 images와 labels annotation 생성. 
# make_yolo_anno_file(val_df, '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/images/val/', '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/labels/val/')
```

<br>

# 3. Oxford Pet Dataset 학습 수행.

- 생성된 Directory 구조에 맞춰서 dataset용 yaml 파일 생성.



```python
# !wget -O '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/ox_pet.yaml' https://raw.githubusercontent.com/chulminkw/DLCV/master/data/util/ox_pet.yaml
```

    --2022-01-12 08:58:57--  https://raw.githubusercontent.com/chulminkw/DLCV/master/data/util/ox_pet.yaml
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.110.133, 185.199.111.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 754 [text/plain]
    Saving to: ‘/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/ox_pet.yaml’
    
    /content/drive/MyDr 100%[===================>]     754  --.-KB/s    in 0s      
    
    2022-01-12 08:58:57 (32.0 MB/s) - ‘/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/ox_pet.yaml’ saved [754/754]
    
    


```python
!mkdir '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ultra_workdir'
```


```python
###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. 
!cd /content/yolov3; python train.py --img 640 --batch 16 --epochs 20 --data '/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ox_pet/ox_pet.yaml' --weights yolov3.pt --project='/content/drive/MyDrive/딥러닝/Computer Vision/Image/OXfordpet/ultra_workdir' \
                                     --name pet --exist-ok
```
