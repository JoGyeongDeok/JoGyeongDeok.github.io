---
title: "Attention Is All You Need (1)"
tags: [Pytorch, Deep Learning, Attention, Transformer]
comments: true
excerpt: Transformer
date : 2023-02-07
categories: 
  - PaperReview
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : NLP
---

 |   |   |
|:-:|:---|
|**저자**|Ashish Vaswani*, Noam Shazeer*, Niki Parmar∗, Jakob Uszkoreit∗, <br>Llion Jones∗, Aidan N. Gomez∗, Łukasz Kaiser∗, Illia Polosukhin∗|
|**게재일자**||
|**요약**|최근 Sequence transduction model들은 복잡한 순환 신경망,<br> convolutional 신경망을 포함하는 Attention Mechanism의 <br> Encoder, Decoder에  의존하고 있다. <br>본 논문에서는 복잡한 RNN, CNN을 사용하지 않는<br> Attention Mechanism을 기초하는 Transformer를 소개한다.|

# **I. Introdution**

RNN, LSTM 그리고 GRU은 language model과 기계번역 같은 sequence 모델에 대한 접근법들로 단단히 자리잡아 왔다. 그 이후로 순환 모델과 encoder-decoder 기술들의 경계를 넓히기 위해 수많은 노력이 계속되고 있다.
<br>

하지만 순환 모델들의 경우 sequence의 hidden state $h_t$를 생성하기 위해 이전 hidden state $h_{t-1}$와 $t$시점의 input을 사용해야한다.
<br>

이러한 순차적 특성들은 훈련 시 병렬화를 할 수 없게 되며, 메모리 한계로 인해 batch cross가 제한되기 된다. 따라서 더 긴 시퀀스 길이에서 중요한 문제가 된다.
<br><br>

본 논문에서는 Transformer를 사용하여 **recurrence**를 피하고 전체적으로 **Attention Mechanism**을 사용하여 이 문제를 해결한다.

# **II. Model Architecture**

<img src = 'https://drive.google.com/uc?id=17fVIeN0bbZ5Bs6cJmZCUlCN7sT4jQ2tc' height = 500 width = 400>

Transformer는 기본적으로 **Encoder**와 **Decoder**로 이루어져 있다. Encoder에서는 input $(x_1,...,x_n)$을 continuous representations $\textbf{z}\ =\ (z_1,...,z_n)$으로 매핑한다. 이후 decoder를 거쳐 output sequence $(y_1,...,y_n)$을 출력한다.

## **2.1 Encoder and Decoder Stacks**

**Encoder** : Encoder는 N(본 논문에서는 6)개의 독립적인 layer들로 구성되어 있고, 각 layer는 2개의 sub-layer로 구성되어 있다. 두 sub-layer 모두 residual connection와 Normalization을 사용한다.
 - Multi-head self attention mechanism
 - Position-wise fully connected feed-forward network

**Decoder** : Decoder는 N(본 논문에서는 6)개의 독립적인 layer들로 구성되어 있고, Encoder의 sub-layers 뿐만 아니라 세 번째 sub-layer까지 포함되어 있다. 이 세 번째 sub-layer는 Encoder stack의 출력에 대해 Multi-head attention을 수행한다.이후 Encoderdhk 마찬가지로 residual connection와 Normalization을 사용한다. 

## **2.2 Attention**

Attention 함수는 query와 key-value set을 출력에 매핑할 수 있다. 여기서 query, key, value는 모두 vector이다. 그리고 출력은 value들의 가중합으로 계산된다.

<img src = 'https://drive.google.com/uc?id=1tlBT3pU98ZNvyZ0WbL-_YQiCe9Hc9bbK' height = 300 width = 500>

**Scaled Dot-Product Attention** : 차원이 $d_k$인 Q(query)와 K(key), 차원이 $d_v$인 V(value)들로 구성되어 있다. 따라서 $QK^T$를 $d_k$로 나누어 표준하고 softmax 함수를 취한 다음 V와 행렬곱을 수행한다.

$$
Attention(Q,\ K,\ V)\ =\ softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

<br>

**Multi-Head Attention** : Q(query), K(key), V(value)에 single attention를 수행하기 전에 각각 $d_k,\ d_k,\ d_v$의 차원으로 linearly project한 후에 Attention function을 사용한다. 이 과정을 h번 반복한다. 이후 Concatenated 하고, 다시 한번 linearly project를 시행한다. 본 논문에서는 $h=8,\ d_k\ =\ d_v\ =\ d_{model}/h\ =\ 64$를 사용하였다.

$$
MultiHead(Q,\ K,\ V)\ =\ Concat(head_1,...,head_h)W^O\\
\ \ \ \ \ where,\ head_i\ =\ Attention(QW_i^Q,\ KW_i^K,\ VW_i^V)
$$
$W_i^Q\ \in\ \mathbb{R}^{d_{model}\times d_k},\ W_i^K\ \in\ \mathbb{R}^{d_{model}\times d_k},\ W_i^V\ \in\ \mathbb{R}^{d_{model}\times d_v},\ W^O\ \in\ \mathbb{R}^{hd_v \times d_{model}}$ 


## **2.3 Position wise Feed-Forward Networks**

sub-layers 이외에도, Encoder와 Decoder의 각 layer에는 **fully connected feed-forward network**도 포함된다. 두 개의 선형 변환 함수와 ReLU 활성화 함수를 사용한다.
$$
FFN(x)\ =\ max(0,\ xW_1\ +\ b_1)W_2\ +\ b_2
$$


## **2.4 Positional Encoding**

본 모델은 순환, 합성곱 신경망 모두 사용하지 않기 때문에 Sequence의 순서를 알기 위해서는, Sequence의 상대적 또는 절대적 위치에 대한 정보를 추가해야 한다. 따라서 **Positional Encodings**을 Embedding 끝에 자리잡아 위치 정보를 추가한다. 이 작업에서 $sin$, $cos$ 함수를 사용하였다.

$$
PE_{(pos,2i)}\ =\ sin(pos/10000^{2i/d_{model}})\\ 
PE_{(pos,2i+1)}\ =\ cos(pos/10000^{2i/d_{model}})
$$
$pos$는 position을 뜻하고, $i$는 차원을 뜻한다.


## **3. Why Self-Attention**

 |   |   |   |   |
|:---|:---|:---|:---|
|**Layer Type**|**Complexity per Layer**|**Sequential Operations**|**Maximum Path Length**|
|Self-Attention|$O(n^2\cdot d)$|$O(1)$|$O(1)$|
|Recurrent|$O(n\cdot d^2)$|$O(n)$|$O(n)$|
|Convolutional|$O(k \cdot n \cdot d^2)$|$O(1)$|$O(log_k(n))$|
|Self-Attention(restricted|$O(r \cdot n\cdot d)$|$O(1)$|$O(n/r)$|

본 논문은 왜 순환, 합성곱 신경망이 아니라 Self-Attention layer들을 사용해야 하는지 서술하고 있다. 총 세가지 이유에서 내용을 주장한다.

 - 계층당 총 계산 복잡성
  - Sequence 길이 $n$이 representation demenssion $d$보다 작을 때 Self-Attention Layer가 Reccurent Layer보다 빠르다. 이는 word-piece와 byte-pair와 같은 기계 번역에서 사용되는 문장 표현이 가장 일반적이다. 매우 긴 Sequence를 포함하는 작업에 대해 계산 성능을 향상시키기 위해, 각 출력 위치를 중심으로 한 입력 Sequence에서 크기 $r$의 근방만 고려하는 것으로 Self-Attention을 제한할 수 있다.(이 방법에 대해서는 향후에 더 접근할 것이라 설명한다.)
 - 최소 순차 연산 수로 측정할 때의 병렬화 가능한 연산량
  - Self-Attention Layer는 모든 위치를 일정한 수의 순차적으로 실행되는 연산으로 연결되는 반면, Recurrent Layer는 $O(n)$ 순차 연산이 필요하다.
 - 네트워크에서 long-range dependencies 사이의 경로 길이
  - long-range dependencies을 학습하는 것은 다수의 Sequence 변환 작업에서 중요한 과제이다. 즉 forward와 backward 사이에 신호가 짧을수록 long-range dependencies를 학습하기 더 쉽다. 따라서 다른 계층 유형으로 구성된 네트워크에서 두 입력 및 출력 위치 사이의 최대 경로 길이를 비교한다.

