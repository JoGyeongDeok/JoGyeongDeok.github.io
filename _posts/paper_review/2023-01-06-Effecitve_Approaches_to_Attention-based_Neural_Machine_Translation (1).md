---
title: "Effecitve Approaches to Attention-based Neural Machine Translation (1)"
tags: [Pytorch, Deep Learning, Attention]
comments: true
excerpt: Attention
date : 2023-01-06
categories: 
  - paper_review
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : NLP
---

 |   |   |
|:-:|:---|
|**저자**|Minh-Thang Luong, Hieu Pham, Christopher D.Manning|
|**게재일자**||
|**요약**|Attentional Mechanism은 NMT(Neural Machine Translation)을 개선시키기 위해 <br>최근에 사용되는 기법이다. 번역을 하는 동안에 전체 문장에서 선택적으로 집중한다고<br> 해서 Attention이라는 이름이 붙여졌다. <br><br>본 논문에서는 Global Approach와 Local Approach 크게 두 가지 종류로<br> Attentional Mechanism을 설명한다.|

# **I. Introdution**

Neural Machine Translation(기계변역)은 대규모 번역 작업(English to French, English to German)을 달성했다. 그 모델은 **\<eos\>**가 올 때까지 전체문장을 순차적으로 읽는다. NMT는 한번에 하나의 단어씩 읽으며 종종 종단 간 방식으로 훈련되고 매우 긴 단어 시퀀스로 잘 일반화할 수 있는 기능을 가진 대규모 신경망이다. 즉 거대한 구문 테이블과 언어 모델을 명시적으로 저장할 필요가 없다.
<br><br>

본 논문에서는 단순하고 효과적으로 모델을 디자인 하기 위해 기초적인 두 가지 Attention Based Model을 사용하였다.

# **II. Neural Machine Translation**

<img src = 'https://drive.google.com/uc?id=1WSAl53UOnEZKdOy-oiWdrJNmSeCpMbgl' height = 500 width = 600>

NMT는 조건부확률 $p(y|x)$를 직접적으로 모델링하는 Neural Network이다.
 - Source Sentence : $x_1,\ ...,\ x_n$
 - Target Sentence : $y_1,\ ...,\ y_m$

<br>

NMT는 두가지 요소로 구성된다.
 - **encoder** : 각 source sentence를 대표하는 **s**를 계산. 
 - **decoder** : 한 번에 하나의 대상 단어를 생성하여 조건부 확률을 분해.
 - 본 논문에서는 stacking LSTM Architecture 사용.
 $$log\ p(y|x)\ =\ \sum^{m}_{j=1}\ log\ p(y_j | y_{<j},\ \textbf{s})\\ \ \ \ \ \ \ \ \ \ \ \ \ =\ softmax(g(\textbf{h}_j))$$
 - $g$ : 최종 결과를 vocabulary-sized의 벡터로 출력하도록 변환하는 함수이다.
 - $\textbf{h}_j$ : RNN의 hidden unit. $\ \ \ \textbf{h}_j\ =\ f(\textbf{h}_{j-1},\ \textbf{s})$

 <br>
 
 $\Rightarrow$ **각각의 단어 $y_j$를 해독할 확률을 매개 변수화할 수 있다.**

# **III. Attention-based Models**

## 3**. Attention-based Models**

<img src = 'https://drive.google.com/uc?id=13R2N9inuNGknfwKI12ezBt3PCYEIMhoY' width =600 height = 400>

NMT에서 Attention은 디코더에서 단어를 출력할 때 매 시점마다, 인코더의 전체 문장을 한번 더 사용한다. 이 때 전체 문장을 동일한 가중치로 참고하는 것이 아니라, 해당 시점에서 예측해야할 단어와 연관이 있는 단어를 더 집중(Attention)해서 보게 된다.
 - $$score(\textbf{h}_t,\ \bar{\textbf{h}}_{s})\ =\ \textbf{h}_t^T\bar{\textbf{h}}_{s}$$(Dot-Product Attention)
 - $$a_{ts}\ =\ \frac{exp(score(\textbf{h}_t,\ \bar{\textbf{h}}_s))}{\sum^{S}_{s'=1}exp(score(\textbf{h}_t,\ \bar{\textbf{h}}_{s'})}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ [Attention\ Weights]\ \ score(\textbf{h}_t,\ \bar{\textbf{h}}_{s'})에 \ \ Softmax 적용$$
 -  $c_t\ =\ \sum_{s}a_{ts}\bar{\textbf{h}}_t\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ [Context\ Vector]$
 - $\tilde{h}_t\ =\ f(c_t,\ \textbf{h}_t)\ =\ tanh(\textbf{W}_c[c_t;\textbf{h}_t])\ \ \ \ \ [Attention\ Vector]\ \ W_c$는 학습 가능한 가중치 행렬이고, $[c_t;\textbf{h}_t]$는 concat한다는 의미이다. 
 - $\hat{y}_t\ =\ Softmax(W_y\tilde{h}_t)\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \  \ [최종적인 예측]$

<br>
이 때 $\bar{\textbf{h}}_t$는 인코더에서 $t$ 시점의 Hidden Layer를 뜻하고,  $\textbf{h}_t$는 디코더에서 $t$ 시점의 Hidden Layer를 뜻한다.

## 3-1**. Global Attention**

<img src = 'https://drive.google.com/uc?id=1nayTsdh2tDeCUMkNzm1buWJVHl3i9vJ0' height = 500 width = 500>


 - "Global Attentional Model"은 Context Vector $c_t$를 가져올 때 Encoderd의 모든 Hidden States을 고려한다.
 - Source의 time steps의 수와 동일한 사이즈의 가변길이 Attend Weights $a_t$는 현재 target hidden state $\textbf{h}_t$와 각 source hidden state $\bar{\textbf{h}}_s$와 비교하여 가져온다.

 
|   |   |    |
|:-:|:-:|:-:|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;**이름** &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**스코어 함수** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|**Defined By**|
|$dot$|$score(s_t,\ h_i)\ =\ s_t^T h_i$|Luong et al. (2015)|
|$scaled\ dot$|$score(s_t,\ h_i)\ =\ \frac{s_t^Th_i}{\sqrt{n}}$|Vaswani et al.(2017)|
|$general$|$score(s_t,\ h_i)\ =\ s_t^TW_ah_i$ <br>// $W_a는 학습 가능한 가중치 행렬$|Luong et al.(2015)|
|$concat$|$score(s_t,\ h_i)\ =\ W_a[s_t;h_i]$|Bahdanau et al. (2015)|
|$location-based$|$a_t\ =\ softmax(W_as_t)$<br>$a_t$ 산출시에 $s_t$만 사용하는 방법|Luong et al. (2015)|

## 3-2**. Local Attention**

<img src = 'https://drive.google.com/uc?id=1ag8dpeJz0UwzKNJXNnhBQhIM4vpiff3W' height = 500 width = 500>

"Global Attention"은 각 "Target Word"를 위해 source에 있는 모든 단어들에 집중을 한다. 이는 비싼 연산이고, 단락이나 문서같이 더 긴 문장을 번역 못할 수도 있다. 이를 해결하기 위해 본 논문에서는 **Local Attentional Mechanism**을 제안한다.
<br>

이 접근법은 선택적으로(small window) 문맥에 집중하고 구별이 가능하다. 따라서 비싼 연산을 피할 수 있고, 동일한 시간동안 **Global Attention**보다 학습이 더 쉽다.
<br>

 - 각 **Target Word**에 대한 aligned position $p_t$를 생성한다.
  - $Monotonic\ alignment$**(local-m)** : $p_t=t\ \ \ \ $[source와 target sequences가 단조 증가방향일 때]
  - $Predictive\ alignment$**(local-p)** : $p_t=S \cdot sigmoid(v_p^Ttanh(W_p\textbf{h}_t))$ <br> $W-p,\ v_p$는 학습을 통해 예측되는 model parameters이고,  S는 Sentence의 길이이다. 결국 $p_t\ \in $ [0,S]가 된다.
 - **Context Vector**는 window [$p_t-D,\ p_t+D]$내의 source hidden states에 대한 가중 평균으로 도출된다.(D는 경험적으로 선택된다.)
 - **Global 접근**과 달리, **Local Alignment Vector**는 고정차원이 된다. $\in  \mathbb{R}^{2D+1}$
 - **Gaussian distribution centered**를 위해 최종적으로 <br> $$a_t(s)\ =\ align(\textbf{h}_t,\ \bar{\textbf{h}}_s)\ exp(-\frac{(s-p_t)^2}{2\sigma^2}),$$<br>$\sigma\ =\ \frac{D}{2}\ \ s$ : window 중앙의 $p_t$에 있는 정수,$\ \ p_t$:실수 집합 

### Teacher Forcing work

<img src = 'https://drive.google.com/uc?id=1heP-7mwcVW58WaVgHd75BXZM_jG8cdr2' height = 500 width = 500>