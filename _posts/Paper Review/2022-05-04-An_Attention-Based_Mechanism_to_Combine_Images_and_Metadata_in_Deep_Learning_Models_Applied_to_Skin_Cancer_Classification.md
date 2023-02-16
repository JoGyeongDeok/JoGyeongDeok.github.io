---
title: "An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification"
tags: [Pytorch, Deep Learning, Multi-Modal]
comments: true
excerpt: Multi-Modal
date : 2022-05-04
categories: 
  - PaperReview
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Multi-Modal
---

<br>


***Types of network architecture***

- 본 논문에서 사용되는 구조는 (c) two input layer network이다.


<img src = 'https://drive.google.com/uc?id=164TGj9x4K6Vj-gWb9UebPnx5Lltoc0F1' height = 500 width = 600>
***기존의 image,meta data two inputs의 Neural Network 모델***


<img src = 'https://drive.google.com/uc?id=1WEQ0hTEX37edsKV7bPKR719-CEHg4PyK' height = 500 width = 600>

<br>

# I. COMBINING IMAGE AND METADATA FEATURES


수행 성능과 더욱 로버스트한 methods를 위해 different sources of data로 부터 얻어지거나 다른 알고리즘으로 부터 추출된 different features을 결합하여 학습시킨다.

**Image + meta data**
- 대부분의 방법은 두개의 features를 concatenation 한다.
 > 하지만 Image는 meta data보다 훨씬 고차원이므로 이 방법은 비효율적이다.
- Image를 main source로 meta data를 보조 data로 사용하는 방법이 존재한다.
 > concatenation보다 더 종합적인 접근법으로 수행성능을 개선시킬 수 있다.

<br>

# II. METADATA PROCESSING BLOCK (METABLOCK): AN ATTENTION-BASED MECHANISM TO COMBINE MULTI-SOURCE FEATURES

**어텐션 메커니즘의 정의**

- 인간의 시각적 집중 현상을 구현하기 위한 신경망적 기법


<img src = 'https://drive.google.com/uc?id=1PiAQ-P3_c5Ai7dvn7wZO0slQBKKPZn7i' height = 500 width = 600>
Self-Attention Computer Vision

본 논문에서는 single layer로 구성된 Metadata Processing Block(MetaBlock)을 제안한다.(attention -based mechanism approach)
- Image로부터 추출된 feature maps을 높이기 위해 metadata를 사용한다.
- Block은 LSTM의 Gate으로부터 영감을 받은 접근법이다.
- Single layer neural network, Activation function, Element-wise operation으로 구성된다.


<img src = 'https://drive.google.com/uc?id=1oqzfAyrd8qrKRLHWUlZGae9hckExfmUW' height = 500 width = 600>
$\psi$ : feature 추출기

$\psi_{img}\ =\ g_{cnn}(\bf{x}_{img})$ 

$\psi_{meta}$ : return features in $\mathbb{R^{d_{meta}}}$
 - ex) 성별같은 categorical data를 scalar numbers 로 변환


$d_{meta}$ : dimensionality of the metadata

$$\widetilde{\bf{x}}_{img}\ =\ \psi_{img}(\bf{x}_{img})$, where $\widetilde{\bf{x}}_{img}\ \in\ \mathbb{R}^{k_{img}\times m_{img}\times n_{img}}$$

$\Rightarrow k_{img}\times m_{img}\times n_{img}$ : features map의 숫자와 순서(m x n 크기의 feature map이 k개 존재)이다.

$$\widetilde{\bf{x}}_{meta}\ =\ \psi_{meta}(\bf{x}_{meta})$$

**MetaBlock**

- feature maps $\widetilde{\bf{x}}$를 이끌어 내는 것이 목표이다.
- output feature $\widetilde{\bf{x}}$와 $\widetilde{\bf{x}}_{img}$는 동일한 shape이다.
- Attention mechanism과 유사하게 작동한다.
- metadata 정보를 Image feature maps에 포함하는 것을 통해 more important features를 concentrate 해야한다.


<img src = 'https://drive.google.com/uc?id=1jBkqkoB2JV_p6UOyqw1jp-5FkzeX3BwG' height = 500 width = 600>
$$\widetilde{\bf{x}}\ =\ \sigma [ \tanh [f_b(\widetilde{\bf{x}}_{meta})\ \circ\ \widetilde{\bf{x}}_{img} ]\ +\ g_b(\widetilde{\bf{x}}_{meta}) ]$$

$\sigma$ : 시그모이드 함수(0 ~ 1) 

$\tanh$ : 하이퍼탄젠트함수(-1 ~ 1)

$\Rightarrow$ BatchNomalize와 같은 효과 + LSTM gates 역할
> **scale & shift**

$f_b,\ g_b$ : MetaBlock을 만드는 함수

$\Rightarrow$ single neural network

$$f_b(\widetilde{\bf{x}}_{meta})\ =\ W^T_f \widetilde{\bf{x}}_{meta}\ +\ \bf{w}_{0f}\ \in\ \mathbb{R}^{k_{img}}$$

$$g_b(\widetilde{\bf{x}}_{meta})\ =\ W^T_g \widetilde{\bf{x}}_{meta}\ +\ \bf{w}_{0g}\ \in\ \mathbb{R}^{k_{img}}$$

$\{W_F,\ W_G\}\ \in\ \mathbb{R}^{d_{meta}\times k_{img}}$

$\Rightarrow$ matrices of weights

$$\{\bf{w}_{0f},\ \bf{w}_{0g}\}\ \in\ \mathbb{R}^{k_{img}}$$
$\Rightarrow$ bias


<img src = 'https://drive.google.com/uc?id=1X65eh_IZ9mIoDHHfdIfkt7wUNZE4MLqR' height = 500 width = 600>

<img src = 'https://drive.google.com/uc?id=18vr9ToQaNNXPn1IUPrdJ6t2jOG1KwVdM' height = 500 width = 600>

<img src = 'https://drive.google.com/uc?id=1VVSDaD_EbUx2nL7OIFkia1G3JGYRyXzs' height = 500 width = 600>
<br>
# III. Implement

```python
''' 
구현 
x1 = self.conv1(image)

# attention
x2 = self.Linear1(meta)     #feature map 크기로 신경망 통과.
x3 = self.Linear1(meta)     #feature map 크기로 신경망 통과

# element-wise product를 위해 reshape
x1 = x1.reshape(x1.shape[3], x1.shape[2], x1.shape[1], x1.shape[0])
x2 = x2.reshape(x2.shape[1], x2.shape[0])
x3 = x3.reshape(x3.shape[1], x3.shape[0])

x1 = F.sigmoid(F.tanh(x2 * x1) + x3)    #(batch_size, feature maps, img_m, img_n) 크기로 반환

# 원래대로 돌리기
x1 = x1.reshape(x1.shape[3], x1.shape[2], x1.shape[1], x1.shape[0])
'''
```


```python
x1 = torch.tensor(range(16*64*128*128)).reshape(16, 64, 128, 128)
x2 = torch.tensor(range(16*64)).reshape(16, 64)
x3 = torch.tensor(range(16*64)).reshape(16, 64)
```


```python
x1 = x1.reshape(x1.shape[3], x1.shape[2], x1.shape[1], x1.shape[0])      
x2 = x2.reshape(x2.shape[1], x2.shape[0])        
x3 = x3.reshape(x3.shape[1], x3.shape[0])        
print(x1.shape)
print(x2.shape)
print(x3.shape)
```


```python
x1 = F.sigmoid(F.tanh(x2 * x1) + x3)
print(x1.shape)
```


# Reference
[1] Using photographs and metadata to estimate house prices in South Korea 저자 Changro Lee
