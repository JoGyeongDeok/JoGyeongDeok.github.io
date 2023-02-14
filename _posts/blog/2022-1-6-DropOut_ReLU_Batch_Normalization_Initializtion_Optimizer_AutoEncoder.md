---
title: "DropOut / ReLU / Batch Normalization / Initializtion / Optimizer / AutoEncoder"
category: DeepLearning
tags: [Pytorch, MLP, Deep Learning]
comments: true
date : 2022-01-06
categories: 
  - blog
excerpt: MLP 기법들의 소개
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

MLP에는 크게 3가지 문제가 있다. 
<br>

- **Gradient Vanishing(Exploding) Problem**
> * Layer간의 weight 전파가 반복됨에 따라 gradient(기울기)가 약해지는 현상
> * Sigmoid 함수는 0~1사이의 값으로, 학습이 진행되면서 0으로 수렴할 수 있어서 Back Propagation할 때 weight가 0으로 수렴 할 수 있다.
- **과적합(Over Fitting) 문제**
> * 신경망의 가장 큰 문제점으로 지적되어 온 과적합 문제
- **Internal covariance shift** 
> * 각 layer마다 input 분포가 달라짐에 따라 학습 속도가 느려지는 현상

이를 해결하기 위해 아래와 같은 방법들을 제시한다.
<br><br>


# ReLU (Activation function)


- ReLU : $R(z) = max(0,z)$
> * sparsification이 가능해짐(unit의 output이 0인 노드가 많아진다.)
> * gradient vanishing을 완화시킬 수 있다.
> * 과적합(OverFitting) 방지
> * 미분 계산이 간단하다.
> * 학습이 빠르게 수렴한다.
> * ReLu의 응용버전인 Leaky ReLu, Parametric ReLU, ELU, SeLU가 존재한다.


<img src = 'https://drive.google.com/uc?id=1y-OuayfbdwhkbKqeBgVZ2pEkacSp6VtQ' height = 500 width = 500>

<br><br>

# DropOut

- 학습과정에서 layer의 node을 random하게 drop함으로써, generalization효과를 가진다.(유전자 알고리즘의 아이디어 차용)
- 1epoch 마다 Random하게 확률을 설정하여 drop한다.
- 효율적인 weight 통제 : ReLU + DropOut(sparsicifation)
- train할 때만 DropOut을 사용하고, test, 실제 데이터에 적용할 때는 DropOut을 하지 않는다.


<img src = 'https://drive.google.com/uc?id=1tv7-Dsl3QUYjUCWII_yf_iyWYGTh7_l4' height = 500 width = 600>

<br><br>

# Batch Normalization

- Internal covariance shift
> * 두번째에서 LeLU 적용시 변화 없다.
> * 세번째에서 BatchNormalization 적용 후 ReLu 함수 사용한다.
- Batch Normalization
> * Input 분포를 scale시키고 shift시킴으로써 layer의 분포를 normalization 시킨다.
> * 학습 속도를 향상 시켜주고 gradient vanishing problem도 해결한다.

$$
BN(h;\gamma,\beta)\ =\ \beta \ +\ \gamma \frac{h\ -\ E(h)}{\sqrt{Var(h)\ +\ \epsilon}}
$$
$$
\beta와 \gamma는 BP과정을 통해 학습
$$


<img src = 'https://drive.google.com/uc?id=1qo6aprFMQovL2YfIgbtYu460Sp0xr0YM' height = 500 width = 700>

<br><br>

# Initialization

- 신경망은 처음에 Weigh를 랜덤하게 초기화하고 Loss가 최소호 되는 부분을 찾아간다.
- 이전에는 초기 분포로 Uniform Distribution이나 Normal Distribution을 사용했다.
- Weight를 랜덤하게 초기화 하면 신경망의 초기 Loss값이 달라진다.
> * 즉 신경망을 초기화할 때마다 신경망의 Loss상에서의 위치가 달라질 수 있다.
> * 신경망을 어떻게 초기화하느냐에 따라 학습 속도가 달라진다.
- ***LeCun Initialization*** : LeCun Normal Initialization과 LeCun Uniform Initialization이 있다.
> **LeCun Normal Initialization** : 

$$
W\ \sim\ N(\ 0,Var(\ W\ )\ ) \\ Var(W)\ =\ \sqrt{\frac{1}{n_{in}}}
\\ 여기서\ n_{in}\ :\ 이전\ Layer의\ 노드\ 수
$$
- ***He Initializtion*** : Xavier Initialization은 ReLU 함수를 사용할 때 비효율적이라는 것을 보이는 데, 이를 보완한 초기화 기법이 He Initialization이다.
> 
$$
W\ \sim\ U(\ -\ \sqrt{\frac{1}{n_{in}}},\ +\ \sqrt{\frac{1}{n_{in}}})
$$

<br><br>

# Optimizer

앞서 Batch단위로 Back Propagation하는 과정을 Stochastic Gradient Descent(SGD)하고  이러한 과정을 'Optimization'이라 한다고 설명했습니다. SGD 외에도 SGD의 단점을 보완하기 위한 다양한 Optimizer가 있다. 

**SGD** : 

$$
\theta\ =\ \theta\ -\ \eta\triangledown_{\theta}J(\theta)
$$

**Momentum** : 

$$
v_t\ =\ \gamma\ v_{t-1}\ -\ \eta\triangledown_{\theta}J(\theta) \\ \theta\ =\ \theta\ -\ v_t
$$
> * 미분을 통해 Gradient 방향으로 가되, 일종의 관성을 추가하는 개념이다. 기본적인 SGD에 Momentum을 추가한 수식은 다음과 같습니다. 여기서 Gamma가 Momentum의 파라미터다.

**Nesterov Accelerated Gradient(NAG)** : 

$$
v_t\ =\ \gamma v_{t-1}\ -\ \eta\triangledown_{\theta}J(\theta\ -\ \gamma v_{t-1}) \\ \theta\ =\ \theta\ -\ v_t
$$
> * Momentum으로 이동한 후 Gradient를 구해 이동하는 방식을 수식으로 표현하면 다음과 같다.

**Adaptive Gradient(Adagrad)** : 

$$
G_t\ =\ G_{t-1}\ -\ (\triangledown_{\theta}J(\theta_t))^2 \\ \theta\ =\ \theta\ -\ v_t
$$
> * Adagrad의 개념은 '가보지 않은 곳은 많이 움직이고 가본 곳은 조금씩 움직이자.'

**RMSProp** : 

$$
G\ =\ \gamma G\ +\ (1\ -\ \gamma)(\triangledown_{\theta}J(\theta_t))^2
\\ \theta\ =\ \theta\ -\ \frac{\eta}{\sqrt{G\ +\ \epsilon}}\triangledown_{\theta}J(\theta_t)
$$
> * RMSProp는 Adagrad의 단점을 보완한 방법이다. Adagrad의 단점은 학습이 오래 진행될수록 부분이 계속 증가해 Step Size가 작아진다는 것인데, RMSProp는 G가 무한히 커지지 않도록 지수 평균을 내 계산한다.

**Adaptive Delta(Adadelta)** : 

$$
G\ =\ \gamma G\ +\ (1\ -\ \gamma)(\triangledown_{\theta}J(\theta_t))^2\\ \triangle_{\theta}\ =\ \frac{\sqrt{s\ +\ \epsilon}}{\sqrt{G\ +\ \epsilon}}\ \triangledown_{\theta}J(\theta_t)\\ \theta\ =\ \theta\ -\ \triangle_{\theta}\\ s\ =\ \gamma s\ +\ (1-\gamma)\triangle_{\theta}^2
$$
> * Adagrad의 단점을 보완한 방법이다. Gradient를 구해 움직이는데, Gradient의 양이 너무 적어지면 움직임이 멈출 수 있다. 이를 방지하기 위한 방법이 'Adadelta'이다.

**Adaptive Moment Estimation(Adam)**
> * RMSProp와 Momentum 방식의 특징을 결합한 방법이다.

**Rectified Adam optimizer(RAdam)** 
> * 대부분의 Optimizer는 학습 초기에 Bad Local Optimum에 수렴해 버릴 수 있는 단점이 있다. 학습 초기에 Gradient가 매우 작아져서 학습이 더 이상 일어나지 않는 현상이 발생한다. RAdam은 이러한 Adaptive Learning Rate Term의 분산을 교정하는 Optimizer로, 논문의 저자는 실험 결과를 통해 Learning Rate를 어떻게 조절하든 성능이 비슷하다는 것을 밝혔다.

<br><br>

# Autoencoder

- 분류가 목적이 아니라 feature를 학습하는 것이 목적이다.
- ***입력층 값을 출력층에서 그대로 예측***하는 것을 목적으로 구성된 인공신경망 모형으로 ***데이터 셋을 압축적으로 표현***하여 다층 신경망의 가중치 학습 문제를 해결한다. 
- Y(출력) = X(입력)로 자기 자신을 학습시킨다.
- Autoencoder를 이용하여 학습된 가중치를 다층 인공 신경망의 초긱 가중치로 사용
- 출력층에서 계산해서 
$
\bar{X}를 얻고 error (\hat{X}-X)
$ 계산
- Error가 최소화 되는 방향으로 Back Propagation를 통해 weight update
- MLP형태의 Autoencoder는 잘 쓰이지 않지만 응용이 되서 다른 곳에서 많이 사용한다
> ***-> feature를 추출하는데 의미가 있다.***


<img src = 'https://drive.google.com/uc?id=1GqUKm0CxbhnHd9kxePU8_hIhmgDSHkfC' height = 500 width = 500>

***Hidden layer의 값을 feature로 사용한다.***

<br><br>

## Stacked Autoencoder (지도학습)

* Layer-wise Pre-training
* Layer와 Layer끼리 autoencoder로 학습을 하고 쌓아 올린다.
> * Input data로 autoencoder1을 학습
> * " 1 "에서 학습된 모형의 hidden layer를 input으로 하여 autoencoder2를 학습
> * " 2 "과정을 원하는 만큼 반복
> * " 1~3 "에서 학습된 hidden layer를 쌓아 올림
> * 마지막 Layer를 softmax와 같은 classification 기능이 있는 output layer 추가
> * Fine-tuning으로 전체 다층 신경망을 재학습


<img src = 'https://drive.google.com/uc?id=1p_q4k0zfHPRrsmJE7oA8k4ZYwHJn-w97' height = 500 width = 600>


<img src = 'https://drive.google.com/uc?id=1B91iqVHjIgjFG6mKx315BqqZLgWTB6KB' height = 500 width = 600>

<br><br>

## Denoising Autoencoder

- Good representation ≐ robust feature
- Input data에 nois를 주어 학습 시킴으로써, 어떠한 데이터가 input으로 오든 robust 모형을 만든다.
- "극한 상황에서 훈련을 해야 (training), 실전(test)에 도움이 된다."

> 1) Corruption step 

$$
\tilde{x}\ =\ x\ +\ noise
$$

> 2) Hidden representation 
$$
h\ =\ sigmoid( W\ \tilde{x}+\ b\ )
$$

> 3) Reconstruct 
$$
x'\ =\ W'h\ +\ b
$$

> 4) Minimize error 
$$
x\ -\ x'
$$


<img src = 'https://drive.google.com/uc?id=1ADNdePsCfyZHCOZvAB8kzfpbMwsv5pxa' height = 500 width = 500>


<img src = 'https://drive.google.com/uc?id=15szzKwOj4UVT-WXuLpIKT6AKc2GnxUR1' height = 500 width = 500>

- "Noise가 추가된 X를 원래의 X분포로 끌어들이는 과정"
- ***Stacked Denoising Autoencoder(SDA)*** : SAE에서 AE를 DAE로 대체한 모형 
