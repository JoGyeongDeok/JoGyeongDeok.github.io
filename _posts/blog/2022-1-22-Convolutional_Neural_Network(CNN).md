---
title: "Convolutional Neural Network(CNN)"
tags: [Pytorch, Computer Vision, Deep Learning]
comments: true
date : 2022-01-22
categories: 
  - blog
excerpt: CNN
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

본 내용은 이미지 처리에 자주 사용되는 합성곱 신경망(Convolutional Neural Network)을 구성하는 Layer들에 대한 내용이다.
<br>

<img src = 'https://drive.google.com/uc?id=1hiPzXDuApYCs2kmb6G36loq4FkUntFwK' height = 500 width = 600>

stride : 건너뛰는 넓이

<br>
# Convolutional layer
> receptive field를 정의하여 입력 층의 이미지를  convolutional layer의 노드와 연결하며 입력 이미지의 특징을 추출하는 역할을 담당


<img src = 'https://drive.google.com/uc?id=1WLLMQ52oavFejhagC5IB3Pqvrr0lqftQ' height = 500 width = 600>

<img src = 'https://drive.google.com/uc?id=1a9H7stJzbHLeXGL-L84NUfgpa93Zcc8x' height = 500 width = 600>

stride, filter에 따라 Dimension 감소


- 5 X 5 pixels을 25개의 독립변수로 줄지어 학습보다 주변 pixel과 함께 학습하는 것이 좋다
- filter (receptive field)를 통해 지역(region)의 feature을 뽑아 냄 
- 이미지를 잘 분류 할 수 있도록 "특징(feature)"을 잡아내는 역할

<img src = 'https://drive.google.com/uc?id=1Lxw_iaLo7hmIMDN34t8t-g62hcLNhrwT' height = 500 width = 600>

**Wieght sharing** : 하나의 feature map에 대하여 동일한 weight 값을 적용

---

<img src = 'https://drive.google.com/uc?id=1ACImHIpHdwkphuMAPulB9D7s7XPIWiZu' height = 500 width = 600>

Convolution layer는 Fully-connected layer와 다르게 input image를 그대로 보존한다. 그대로 보존한 input image를 같은 depth를 가진 필터를 통해 쭉 밀어주며 특징을 뽑아준다.

<img src = 'https://drive.google.com/uc?id=1PRrirOfdAPZUElqw6iGRDR8gAHgVBwyz' height = 500 width = 600>

더 구체적으로 말하자면, filter가 W값이 되어, input image 값에 필터 값을 곱해 주고 bias를 더해준 값을 계산하는데 이를 전부 더해주는 방식이다.

<img src = 'https://drive.google.com/uc?id=1IlSIXbCzYplImno6zg40VAH9KIXutIpP' height = 500 width = 600>

이렇게 input image가 필터를 거치게 되면 activation map이 생기게 되는데, 해당 필터에 대해 각 1개가 발생한다. 또, 더 정확한 결과를 위해 필터를 여러개를 사용하는데, 이에 따라 activation maps의 depth는 자연스레 filter의 개수를 따라간다.

<img src = 'https://drive.google.com/uc?id=1P15dsnxi9y1GRT2dEqwVmAPhNOKRszNW' height = 500 width = 600>

<br>

# Pooling layer
> scale invariant, location-invariant한 특징을 발견하는 역할을 수행


<img src = 'https://drive.google.com/uc?id=1Zjpc95RyMKDNYLbwXAh7nABxum8QnHq0' height = 500 width = 600>

- 정보의 손실을 최소화 하면서 차원을 축소시키는 역할을 함
- mean pooling, max poolin 등이 있으며, 주로 max pooling을 사용함
> (모델에 따라 사용 안하기도 한다.)
- 일반적으로 (convolution layer - pooling layer)을 반복하나, 분석가 입장에서 CNN 구조를 디자인 가능

<br>

# Fully connected layer
> 최종적으로 이미지를 분류하는 기능을 수행하는 일반적인 신경망 모형 부분

<img src = 'https://drive.google.com/uc?id=1gedWrkF-r_GgMQXemqUKhQpOzbnFSrHh' height = 500 width = 600>

- 마지막 pooling layer에서 일반적인 다층 신경망(flatten) 구조를 만듦
- 전형적인 CNN 구조 
> [Conv - ReLU - Pool] X N, [FC - ReLU] X M, Softmax (N : 반복 횟수, M : Layer 수)
- 학습 데이터를 늘리기 위해 data augmentation을 사용하기도 한다.

<img src = 'https://drive.google.com/uc?id=1URwip-phLxUUwCqJsoBaiDjFPyioDso9' height = 500 width = 600>


<br>

# CNN 구현


```python
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
```


```python
!wget -O '/audrey02.png' https://upload.wikimedia.org/wikipedia/ko/2/24/Lenna.png
```

    --2022-01-30 09:43:16--  https://upload.wikimedia.org/wikipedia/ko/2/24/Lenna.png
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 103.102.166.240, 2001:df2:e500:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|103.102.166.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 473831 (463K) [image/png]
    Saving to: ‘/audrey02.png’
    
    /audrey02.png       100%[===================>] 462.73K  1.86MB/s    in 0.2s    
    
    2022-01-30 09:43:17 (1.86 MB/s) - ‘/audrey02.png’ saved [473831/473831]
    
    


```python
img = cv2.imread('/audrey02.png')[:,:,1]
cv2_imshow(img)
print(img.shape)
```


    
![png](/images/2022-1-22-Convolutional_Neural_Network%28CNN%29_files/2022-1-22-Convolutional_Neural_Network%28CNN%29_29_0.png)
    


    (512, 512)
    

## Convolutional Layer



```python
filter_size = 3
stride = 2 
```


```python
#Convolutional Layer
def Conv_layer(img, filter_size = 3, stride = 2):
  h = img.shape[0]
  w = img.shape[1]

  # filter 초기 가중치 정의
  filter = np.random.randint(-1,2, (filter_size, filter_size) ) 

  w_range = range(0, w, filter_size + stride)
  h_range = range(0, h, filter_size + stride)

  #Convolved_Feature 생성
  Convolved_Feature_w = int(w/(filter_size + stride))
  Convolved_Feature_h = int(h/(filter_size + stride))
  Convolved_Feature = np.zeros((Convolved_Feature_h, Convolved_Feature_w))

  for i in range(Convolved_Feature_w) : 
    for j in range(Convolved_Feature_h) : 
      filter_img = img[h_range[j]:h_range[j] + filter_size, w_range[i]:w_range[i] + filter_size]
      Convolved_Feature[j,i] = np.sum(filter * filter_img)
  return Convolved_Feature
```


```python
Conv_img = Conv_layer(img)
```


```python
# check image
plt.imshow(Conv_img, cmap='gray', vmin=0, vmax=255)
plt.figure(figsize=(15,15))
plt.show()
```


    
![png](/images/2022-1-22-Convolutional_Neural_Network%28CNN%29_files/2022-1-22-Convolutional_Neural_Network%28CNN%29_34_0.png)
    



    <Figure size 1080x1080 with 0 Axes>


## Pooling Layer


```python
def pooling_layer(img, pooling_size = 2, stride = 2, pooling_type = 'max') : 
  h = img.shape[0]
  w = img.shape[1]

  #Pooling 정의
  w_range = range(0, w, pooling_size + stride)
  h_range = range(0, h, pooling_size + stride)


  #Pooling 생성
  Pooling_w = int(w/(pooling_size + stride))
  Pooling_h = int(h/(pooling_size + stride))
  Pooling_img = np.zeros((Pooling_h, Pooling_w))

  for i in range(Pooling_w) : 
    for j in range(Pooling_h) : 
      Pooling_map = img[h_range[j]:h_range[j] + filter_size, w_range[i]:w_range[i] + filter_size]
      Pooling_img[j,i] = np.max(Pooling_map)
  return Pooling_img
```


```python
Pooling_img = pooling_layer(Conv_img)
```


```python
# check image
plt.imshow(Pooling_img, cmap='gray', vmin=0, vmax=255)
plt.figure(figsize=(15,15))
plt.show()
```


    
![png](/images/2022-1-22-Convolutional_Neural_Network%28CNN%29_files/2022-1-22-Convolutional_Neural_Network%28CNN%29_38_0.png)
    



    <Figure size 1080x1080 with 0 Axes>

