---
title: "Object Detection의 이해"
category: DeepLearning
tags: [Computer Vision, Object Detection, Deep Learning]
comments: true
date : 2022-01-10
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


# **Object Detection & Segmentation 개요**

- Classification : 분류(CNN 등 사용)
- Localization : 단 하나의 Object 위치를 Bounding box로 지정하여 찾음
- Object Detection : 여러 개의 Object들에 대한 위치를 Bounding box로 지정하여 찾음
- Segmentation : Detection보다 더 발전된 형태로 Pixel 레벨 Detection 수행

<img src = 'https://drive.google.com/uc?id=1oVb00iADt5sckCeoWm0KghmOfkEooSsp' height = 500 width = 700>

***Object Detection History***
> - One-stage detector : 바로 Detection 실행 YOLO V1는 빠르지만 정확도가 좋지는 않다. YOLO V2, V3, EfficientDet 등 빠르고 정확도가 높은 Detector가 발전함 

> - Two-stage detector : Object가 존재할만 한 곳을 대략적으로 찾은 후 Detection.실행 정확도가 높지만 실시간 적용이 어렵다(동영상 등) 

<img src = 'https://drive.google.com/uc?id=1hM-SQLlTdG5HTIGOP0GVwKjuwuyCui-s' height = 500 width = 700>
---

***Object Detection의 주요 구성 요소***
- 영역추정(Region Proposal)
> Bounding Box를 예측을 하는 부분(Regression) + 분류(Classification)

- Detection을 위한 Deep Learning 네트웍 구성 
> * Feature Extraction + Classification Layer => Backbone
> * FPN(Neck)
> * Network Prediction(Head) + Classification + Regression

- Detection을 구성하는 기타 요소
> IOU, NMS, mAP, Anchor box

<img src = 'https://drive.google.com/uc?id=1uVdA4jtkDqMP1k0NB4so-knUsKhG_edm' height = 500 width = 700>

작은 Object에 대한 분류를 Neck이 도움, Head에서 분류한다.

<br>

***Object Detection의 난제***

1. Classification + Regression을 동시에
> Loss 함수식이 복잡해진다.
2. 이미지 안에 다양한 크기와 유형의 오브젝트가 섞여 있다.
> 크기가 서로 다르고, 생김새가 다양한 오브젝트가 섞여 있는 이미지에서 이들을 Detect 해야한다.
3. 중요한 Detect 시간
> Detect 시간이 중요한 실시간 영상 기반에서 Detect해야 하는 요구사항 증대
4. 명확하지 않은 이미지
> 오브젝트 이미지가 명확하지 않은 경우가 많다.
5. 데이터 세트의 부족
> 훈련 가능한 데이터 세트가 부족하다. annotation을 만들어야 하기 때문이다.(Bounding Box 수작업해야한다.)


<br>

# 1. Object Localization

 단 하나의 Object 위치를 Bounding box로 지정하여 찾는 것

<img src = 'https://drive.google.com/uc?id=18xy73zrVIq7BBUyTyPi6YCTQDGjOOi90' height = 500 width = 600>
Annotation -> Bounding Box 좌표

<img src = 'https://drive.google.com/uc?id=1m6WWLBTJUQryCbS4GxPaRTfZUClgmsJL' height = 500 width = 600>
두개 이상의 Object 검출에서 Object가 존재할 만한 위치(Region Proposal), NN, 예측 해야한다.

<br>

# 2. Object Detection

## ***Sliding Window 방식***



Window를 왼쪽 상단에서 오른쪽 하단으로 이동시키면서 Object Detection하는 방식 
> **Window의 크기, 위치가 적합하지 않는 경우가 존재**
> * 1. 다양한 형태의 Window를 각각 sliding 시키는 방식
> * 2. Window Scale은 고정하고 scale을 변경한 이미지를 사용하는 방식
> * Object Detection의 초기 기법으로 활용 

- 오브젝트 없는 영역도 무조건 슬라이딩 하여야 하며 여러 형태의 Window와 여러 Scale을 가진 이미지를 스캔해서 검출해야 하므로 수행시간이 오래 걸리고 검출 성능이 상대적으로 낮다
- Region Proposal기법의 등장으로 활용도는 떨어졌지만 Object Detection 발전을 위한 기술적 토대 제공

<img src = 'https://drive.google.com/uc?id=1nxT7PZhOPfiElMID6JoffEYx9HbOaLNa' height = 500 width = 600>

## ***Region Proposal(영역 추정) 방식***

### ***Selective Search 기법***

- 빠른 Detection과 높은 Recall 예측 성능을 동시에 만족하는 알고리즘
- 각각의 object들이 1개의 개별 영역에 담길 수 있도록 많은 초기 영역을 생성
- 컬러, 무늬, 크기, 형태에 따라 유사한 Region을 계층적 그룹핑 방법으로 계산

<img src = 'https://drive.google.com/uc?id=138VQOLry4bS4I75PV7kM-Txt7W1gV9Ff' height = 500 width = 700>
1. 개별 Segment된 모든 부분들을 Bounding box로 만들어서 Region Proposal 리스트로 추가
2. 컬러, 무늬, 크기, 형태에 따라 유사도가 비슷한 Segment들을 그룹핑함
3. 다시 1번 Step Region Proposal 리스트 추가, 2번 Step 유사도가 비슷한 Segment들 그룹핑을 계속 반복 하면서 Region Proposal을 수행

#### Selective Search 실습


```python
!pip install selectivesearch
```

    Collecting selectivesearch
      Downloading selectivesearch-0.4.tar.gz (3.8 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from selectivesearch) (1.19.5)
    Requirement already satisfied: scikit-image in /usr/local/lib/python3.7/dist-packages (from selectivesearch) (0.18.3)
    Requirement already satisfied: scipy>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (1.4.1)
    Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (1.2.0)
    Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (3.2.2)
    Requirement already satisfied: pillow!=7.1.0,!=7.1.1,>=4.3.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (7.1.2)
    Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (2021.11.2)
    Requirement already satisfied: networkx>=2.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (2.6.3)
    Requirement already satisfied: imageio>=2.3.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->selectivesearch) (2.4.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->selectivesearch) (1.3.2)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->selectivesearch) (2.8.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->selectivesearch) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->selectivesearch) (3.0.6)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib!=3.0.0,>=2.0.0->scikit-image->selectivesearch) (1.15.0)
    Building wheels for collected packages: selectivesearch
      Building wheel for selectivesearch (setup.py) ... [?25l[?25hdone
      Created wheel for selectivesearch: filename=selectivesearch-0.4-py3-none-any.whl size=4350 sha256=24d44c7330b29e1f77c34f6558f392029a4caa17e56a31f8deb2666aa1a5983f
      Stored in directory: /root/.cache/pip/wheels/83/0e/c9/4713ec9c1692e688f84fd3e80201018a02992949ca63697ba8
    Successfully built selectivesearch
    Installing collected packages: selectivesearch
    Successfully installed selectivesearch-0.4
    


```python
path = '/content/drive/MyDrive/딥러닝/Computer Vision/Image/'
!mkdir '/content/drive/MyDrive/딥러닝/Computer Vision/Image'
!wget -O '/content/drive/MyDrive/딥러닝/Computer Vision/Image/audrey01.jpg' https://raw.githubusercontent.com/chulminkw/DLCV/master/data/image/audrey01.jpg
```


```python
import numpy as np
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import os
%matplotlib inline

## 오드리햅번 이미지를 cv2로 로드하고 matplotlib으로 시각화
img = cv2.imread(path+'/audrey01.jpg') 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape : ', img.shape)

plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
plt.show()
```

    img shape :  (450, 375, 3)
    


    
![png](/images/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_files/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_29_1.png)
    



```python
_, regions = selectivesearch.selective_search(img_rgb, scale = 100, min_size = 2000) # scale : 크게하면 큰 사이즈의 Object , min_size : 적어도 2000의 Object 추천
print(type(regions), len(regions))
```

    <class 'list'> 41
    

#### 반환된 Region Proposal(후보 영역)에 대한 정보 보기

반환된 regions 변수는 리스트 타입으로 세부 원소로 딕셔너리를 가지고 있다. 개별 딕셔너리 내 KEY값 별 의미
- rect : x, y 시작 좌표, 너비, 높이 값을 가지며 이 값이 Detected Object 후보를 나타내는 Bounding box이다.
- size : Object의 크기
- labels : 해당 rect로 지정된 Bounding Box 내에 있는 오브젝트들의 고유 ID
- 아래로 내려갈 수록 너비와 높이 값이 큰 Bounding Box이며 하나의 Bounding box에 여러개의 오브젝트가 있을 확률이 커진다.


```python
regions[30:-3] #labels가 여러개면 유사한 labels들은 합친다는 뜻이다.
```




    [{'labels': [14.0, 16.0], 'rect': (0, 253, 92, 191), 'size': 12201},
     {'labels': [14.0, 16.0, 7.0, 11.0, 9.0],
      'rect': (0, 91, 183, 353),
      'size': 26876},
     {'labels': [10.0, 15.0, 19.0], 'rect': (0, 171, 326, 278), 'size': 52532},
     {'labels': [10.0, 15.0, 19.0, 8.0, 13.0],
      'rect': (0, 97, 374, 352),
      'size': 66436},
     {'labels': [17.0, 18.0], 'rect': (84, 312, 100, 115), 'size': 6357},
     {'labels': [17.0, 18.0, 14.0, 16.0, 7.0, 11.0, 9.0],
      'rect': (0, 91, 184, 353),
      'size': 33233},
     {'labels': [17.0, 18.0, 14.0, 16.0, 7.0, 11.0, 9.0, 12.0],
      'rect': (0, 91, 195, 353),
      'size': 36101},
     {'labels': [17.0, 18.0, 14.0, 16.0, 7.0, 11.0, 9.0, 12.0, 2.0, 6.0],
      'rect': (0, 0, 374, 444),
      'size': 61244}]



**Bounding Box를 시각화 하기**




```python
cand_rects = [cand['rect'] for cand in regions]
green_rgb = (125, 255, 51)
img_rgb_copy = img_rgb.copy()
for rect in cand_rects:

  left = rect[0]
  top = rect[1]
  right = left + rect[2]
  bottom = top + rect[3]

  img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color = green_rgb, thickness = 2) # 사각형 그리는 모듈
plt.figure(figsize=(8,8))
plt.imshow(img_rgb_copy)
plt.show()
```


    
![png](/images/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_files/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_35_0.png)
    


Bounding Box의 크기가 큰 후보만 추출



```python
cand_rects = [cand['rect'] for cand in regions if cand['size']>30000]
green_rgb = (125, 255, 51)
img_rgb_copy = img_rgb.copy()
for rect in cand_rects:

  left = rect[0]
  top = rect[1]
  right = left + rect[2]
  bottom = top + rect[3]

  img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color = green_rgb, thickness = 2)
plt.figure(figsize=(8,8))
plt.imshow(img_rgb_copy)
plt.show()
```


    
![png](/images/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_files/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_37_0.png)
    


# 3. Object Detection 성능 평가 Metric - IoU

**IoU : Intersection over Union**
> 모델이 예측한 결과와 실측(Box)가 얼마나 정확하게 겹치는가를 나타내는 지표
$$
IoU = \frac{Area\ of\ Overlap\ (\ 개별\ Box가\ 서로\ 겹치는\ 영역\ )}{Area\ of\ Union\ (\ 전체\ Box의\ 합집합\ 영역\ )}
$$

<img src = 'https://drive.google.com/uc?id=1I8j7OwFZnHLaHbkfxNV_eeA4ZugDg8cn' height = 500 width = 600>

## IoU 실습


```python
def coumpute_iou(cand_box, gt_box):
  
  # Calculate intersection areas
  x1 = np.maximum(cand_box[0], gt_box[0])
  y1 = np.maximum(cand_box[1], gt_box[1])
  x2 = np.minimum(cand_box[2], gt_box[2])
  y2 = np.minimum(cand_box[3], gt_box[3])

  intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
  
  cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
  gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
  union = cand_box_area + gt_box_area - intersection

  iou = intersection / union
  return iou
```


```python
# 실제 box(Ground Truth)의 좌표를 아래와 같다고 가정.
gt_box = [60, 15, 320, 420]

img = cv2.imread(path+'/audrey01.jpg') 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

red = (255, 0, 0)
img_rgb = cv2.rectangle(img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color = red, thickness = 2)
plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
plt.show()
```


    
![png](/images/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_files/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_43_0.png)
    



```python
cand_rects = [cand['rect'] for cand in regions if cand['size'] > 3000]
img_rgb = cv2.rectangle(img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color = red, thickness = 2)

for index, cand_box in enumerate(cand_rects):
  cand_box = list(cand_box)
  cand_box[2] += cand_box[0]
  cand_box[3] += cand_box[1]

  iou = coumpute_iou(cand_box, gt_box)

  if iou > 0.5:
    print('index : ', index, 'iou : ', iou, 'rectangle : ',(cand_box[0], cand_box[1], cand_box[2], cand_box[3]))
    cv2.rectangle(img_rgb, (cand_box[0], cand_box[1]), (cand_box[2], cand_box[3]), color = green_rgb, thickness = 1)
    text = "{}: {:.2f}".format(index, iou)
    cv2.putText(img_rgb, text, (cand_box[0] +100, cand_box[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color = green_rgb, thickness = 1)
plt.figure(figsize = (12,12))
plt.imshow(img_rgb)
plt.show()

```

    index :  8 iou :  0.5184766640298338 rectangle :  (72, 171, 324, 393)
    index :  18 iou :  0.5409250175192712 rectangle :  (72, 171, 326, 449)
    index :  28 iou :  0.5490037131949166 rectangle :  (0, 97, 374, 449)
    index :  32 iou :  0.6341234282410753 rectangle :  (0, 0, 374, 444)
    index :  33 iou :  0.6270619201314865 rectangle :  (0, 0, 374, 449)
    index :  34 iou :  0.6270619201314865 rectangle :  (0, 0, 374, 449)
    index :  35 iou :  0.6270619201314865 rectangle :  (0, 0, 374, 449)
    


    
![png](/images/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_files/2022-1-10-Object_Detection%EC%9D%98_%EC%9D%B4%ED%95%B4_44_1.png)
    


## ***NMS(Non Max Suppression)***

- Object Detection 알고리즘은 Object가 있을 만한 위치에 많은 Detection을 수행하는 경향이 강하다.
- NMS는 Detected된 Object의 Bounding box중에 비슷한 위치에 있는 box를 제거하고 가장 적합한 box를 선택하는 기법

<img src = 'https://drive.google.com/uc?id=1hMB7J_syXwLx7xCjKJmXFQSGW3iT4_5g' height = 500 width = 600>
1. Detected 된 Bounding Box 별로 특정 Confidence threshold 이하 Bounding Box는 먼저 제거(confidence score < 0.5)
2. 가장 높은 confidence score를 가진 box 순으로 내림차순 정렬하고 아래 로직을 모든 box에 순차적으로 적용
> 높은 confidence score를 가진 box와 ***겹치는*** 다른 box를 모두 조사하여 IoU가 특정 threshold 이상인 box를 모두 제거(IoU Threshold > 0.4)
3. 남아 있는 Box만 선택

* Confidence Score가 ***높을 수록***
* IoU Threshold가 ***낮을 수록*** 많은 Box가 제거됨.

<img src = 'https://drive.google.com/uc?id=17OulonFcT-R_CymXuzvyeZLZ08bmC5eH' height = 500 width = 600>

## ***Object Detection 성능평가 Metric - mAP(mean Average Precision)***

실제 Object Detection된 재현율(Recall)의 변화에 따른 정밀도(Presion)의 값을 평균한 성능 수치

- IoU
- Precision-Recall Curve, Average Precision (AUC)
- Confidence threshold

<img src = 'https://drive.google.com/uc?id=1O01ZKa9CQgavS660FVSHHgvwHHBoabgV' height = 500 width = 600>
<img src = 'https://drive.google.com/uc?id=1bZCxVnCBxE96_kLzkfup_j49hztg9TAQ' height = 500 width = 600>
<img src = 'https://drive.google.com/uc?id=1VLV412Zi0rJNT4IDLN4J6LUlDu0cJBuu' height = 500 width = 600>
- AP는 한 개 오브젝트에 대한 성능 수치
- mAP(mean Average Precision)은 여러 오브젝트들의 AP를 평균한 값

**COCO Challenge에서의 mAP** 
- 예측 성공을 위한 IoU를 0.5 이상으로 고정한 PASCAL VOC와 달리 COCO Challenge는 IoU를 다양한 범위로 설정하여 예측 성공 기준을 정함.
- IoU를 0.5 부터 0.05씩 값을 증가시켜 0.95 까지 해당하는 IoU별로 mAP를 계산
- 또는 크기의 유형(대/중/소)에 따른 mAP도 측정

<img src = 'https://drive.google.com/uc?id=1fNMgQAkMxsz8Wjhp2BHbSsTej7qzuCzL' height = 500 width = 600>
 - **AP는 0.95 IoU가 0.95 일 때이다.**
