---
title: "GAN(Generative Adversarial Networks)"
tags: [Pytorch, Computer Vision, Deep Learning]
comments: true
date : 2022-02-16
categories: 
  - blog
excerpt: GAN
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

<br>

<img src = 'https://drive.google.com/uc?id=1vikQdYF5MSaXBqoqOK2eZpB-oSn4k8uk' height = 500 width = 700>

GAN을 처음 제안한 Ian Goodfellow는 <경찰과 위조지폐범>으로 비유하였다. 

지폐위조범이 처음에는 돈을 제대로 못 만들어 경찰이 위조지폐를 제대로 구분하여 검거에 성공했다. 이후, 지폐위조범은 더욱 발전된 기술로 지폐를 위조한다. 위조지폐범은 진짜 같은 위조지폐를 만들어(생성, gnerater) 경찰을 속이고, 경찰은 진짜와 가짜 화폐를 구분(분류, discriminator)하기 노력한다.

​

결국 위조지폐범은 구분하기 어려운 위조지폐를 만들게 된다. 경찰은 이게 진짜인지 가짜인지 구별하기 가장 어려운 50% 확률에 수렴하게 된다.

[출처] GAN(Generative Adversarial Networks) 논문 리뷰|작성자 흐이준

---

***Discriminator의 학습 과정*** :

Discriminator의 역할은 진짜 데이터를 진짜로, 가짜 데이터를 가짜로 분류하는 것이다. 그렇기 때문에 Input으로서 가짜/진짜 데이터가 모두 필요합니다.
Input Noise를 G에 넣어 가짜 데이터를 만들고 진짜 데이터를 각각 D에 넣어 가짜는 0, 진짜는 1로 Label을 설정해 학습을 진행


<img src = 'https://drive.google.com/uc?id=1RoLwzucJxsn8gKTS5gryqfmfgDBOPrl1' height = 500 width = 700>

***Generator의 학습 과정*** : 

Gan의 최종 목적은 데이터를 '생성'해내는 것이기 때문에 Generator를 학습시키는 것입니다. Generator G를 잘 학습시키기 위해서는 Discriminator D를 잘 속여야 합니다. 먼저 Noise로부터 G를 통해 가짜 데이터를 만들고 이를 D에 Input으로 넣습니다. 여기까지가 Feed Forward의 과정입니다.

일반적인 Neural Network의 Feed Forward를 생각해 보면 약간 헷갈릴 수 있습니다. 일반적인 Neural Network는 Input과 Hidden Layer를 거쳐 Output을 계산하는 것까지를 Feed Forward라 하는데, Generator는 Generator의 Output을 다시 Discriminator의 Input으로 넣어 Output까지 계산해야 합니다. G의 목적이 D를 속이는 것이고 D의 에러를 통해 G의 Back Propagation을 계산하기 때문에 G의 Feed Forward는 D까지 거치게 되는 것입니다.

<img src = 'https://drive.google.com/uc?id=1ppQ3k9hESbOSaAZneeu2Ibw5swUdp6dm' height = 500 width = 700>


<br>

# Minmax Problem of GAN 

$$
min_G max_D V(D,\ G)\ =
\ E_{x\sim p_{data}(x)}[log\ D(X)]\ +
\ E_{z\sim p_z(z)}[log\ (1\ -\ D(G(z)))]
$$

$Q_{model}(x|z) $ : 정의하고자 하는 z값을 줬을 때 x 이미지를 내보내는 모델


$P_{data(x)}$ : x라는 data distribution은 있지만 어떻게 생긴지 모르므로, P 모델을 Q 모델에 가깝게 가도록 함

$x$ : real image

$z$ : latent code

$G(Z)$ : fake image

$D(x)$ : real image라고 분류한 확률

$D(G(z))$ : D가 fake라고 분류한 확률

> $G(z)$는 D(G(z))가 1로 판단하도록 학습하고,
$D(G(z))$는 0으로 판단하도록 학습함

$\color{blue} {파란 점선}$ : 분류 분포 
- 학습을 반복하다보면 가장 구분하기 어려운 구별 확률인 $\frac{1}{2}$ 상태가 된다.

$\color{green} {초록 점선}$ : 가짜 데이터 분포

$\color{black} {검은색 점선}$ : 실제 데이터 분포


<img src = 'https://drive.google.com/uc?id=1Eb1HHd66uI-nqkFi0ym9PAOOmmrKxsUv' height = 500 width = 700>

|   |   |
|:---|:---|
|D should maximize V(D, G): <br><br> D 입장에서 V가 최댓값|1. D가 구분을 잘하는 경우, 만약 Real data가 들어오면 <br> $D(x)\ =\ 1,\ D(G(z))\ =\ 0$ : <br> 진짜면 1, 가짜면 0을 내뱉음. $(G(z)$)에 가짜가 들어온 경우, <br> 가짜를 잘 구분한 것이다.) <br>- D의 입장에서는 $minmaxV(D, G)\ =\ 0$<br>- Maxmize를 위해 0으로 보내는 것이 D의 입장에서는 가장 좋다.|
|D should minimize V(D,G): <br><br> G입장에서 V가 최솟값|2. D가 구분을 못하는 경우, 만약 Real data가 들어온다면<br>$D(G(z)) = 1$ : <br>진짜를 0, 가짜를 1로 내뱉음<br>(진짜를 구분하지 못하고 가짜를 진짜로 착각한다.)<br>- log안의 D 값이 0이 되어, V 값이 $-\infty$ 가 된다.<br>- Minimize를 위해 $-\infty$로 보내는 것이 G의 입장에서는 가장 좋다.|

<br>

# Advantages and Disadventages

## Advantages

- 기존의 어떤 방법보다도 사실적인 결과물 생성 가능
- 데이터의 분포를 학습하기 때문에, 확률 모델을 명확하게 정의하지 않아도 Generator가 샘플(fake)을 생성 가능
- ***MCMC(Markov Chain Monte Carlo)***를 사용하지 않고, backprop을 이용하여 gradient를 구할 수 있다.
- 네트워크가 매우 샤프하고 변형된 분포 표현 가능

*Markov Chain : 시간에 따른 계의 상태의 변화를 나타내며, 매 시간마다 계는 상태를 바꾸거나 같은 상태를 유지하며 상태의 변화를 '전이'라 합니다. 마르코프 성질은 과거와 현재 상태가 주어졌을 때의 미래 상태의 조건부 확률 분포가 과거 상태와는 독립적으로 현재 상태에 의해서만 결정된다는 것을 뜻합니다.*

​

*Monte Carlo : 난수를 이용하여 함수의 값을 확률적으로 계산하는 알고리즘을 부르는 용어이다. 수학이나 물리학 등에 자주 사용됩니다. 이 방법은 계산하려는 값이 닫힌 형식(closed form)으로 표현되지 않거나 매우 복잡한 경우에 확율/근사적으로 계산할 때 사용되며 유명한 도박의 도시 몬테카를로의 이름을 본따 만들어 졌습니다.*

출처 : youtu.be/soyoJqIbW-A, youtu.be/zhRCwtOO3Fg

## Difficulties & Disadvantage

- Simple Example : unstable
> Minmax 최적화 문제를 해결하는 과정이기 때문에, oscillation이 발생하여 모델이 불안정할 수 있다.
> * 두 모델의 성능 차이가 있으면 한쪽으로 치우치기 때문에, **DCGAN(Deep Convolution GAN)**으로 문제 개선

- Minibatch Discrimination
> 컨버전스가 안되는 경우는 여러 케이스를 보여준다.

- Ask Somebody
> 모호한 평가 기준
> * Inception score 사용하여 문제 해결

- Mode Collapse (sample diversity)
> - G가 '어떤 특정 데이터를 만들더니 D가 속더라' 라는 것을 알게 된다면, G의 입장에서는 그 특정 데이터만 만들려고 노력한다.
> - 학습 데이터의 분포를 따라가기 때문에 어떤 데이터가 생성될지 예측하기 어렵다.
> *  cGAN(Conditional GAN)을 이용하여 문제 개선
