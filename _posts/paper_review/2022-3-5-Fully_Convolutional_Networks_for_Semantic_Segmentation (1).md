---
title: "Fully Convolutional Networks for Semantic Segmentation (1)"
tags: [Pytorch, Deep Learning, Computer Vision, Semantic Segmentation]
comments: true
excerpt: Semantic Segmentation
date : 2022-03-05
categories: 
  - paper_review
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Computer Vision
---
<style>
      .language-plainte{
    background-color : rgba($color: #ffffff, $alpha: 1.0);
    margin-bottom: 3em;
  }
  .scrol{
    height : 700px;
    overflow:auto;
  }

</style>


<br>

|   |   |
|:-:|:---|
|**저자**|NIZAM UD DIN, KAMRAN JAVED, SEHO BAE, AND JUNEHO YI|
|**게재일자**|2017 Apr;39(4):640-651. doi: 10.1109/TPAMI.2016.2572683. Epub 2016 May 24.|
|**요약**|기존의 classification network구조를 이용하여 semantic segmentation 문제를 <br>해결하려고 했었다. 하지만 fully connected layer에서 기존의 3D feature를 <br>일렬로 늘어놓아 1D 벡터로 만들어 연산하기 때문에 위치에 대한 정보를 <br>잃어버리게 된다. 이러한 단점들을 보완하기 위해 FC(fully connected) layer를<br> 제거하고 모든 layer를 convolution layer(합성곱층)로만 구성하여 semantic segmentation<br> 문제를 해결하겠다는 관점이 본 논문의 핵심이다.|


<img src = 'https://drive.google.com/uc?id=1ne6cXq7aPShsqjc6Vw_apboD1cylFZ3Z' height = 500 width = 700>

그 당시 사용되던 대표적인 classification network인 VGG16의 구조

# **I. INTRODUCTION**

Convolutional networks는 image classification, bounding box object etection, keypoint prediction 그리고 local correspondence에서 많은 개선을 이루었다.
<br>

자연스레 다음 순서는 모든 픽셀에 대해 예측하는 하는 것인데,<br>
이전의 연구에서는 각 픽셀은 object혹은 region에 의해 둘러쌓여 클래스로 레이블이 지정되었지만 semantic segmentation을 수행하는 것에서 단점이 있었다.
<br>

따라서 본 논문에서는 FNCs(Fully Convolutional networks)을 소개한다.

# **II. Fully Convolutional Networks**

- 각 레이어의 output은 3차원의 배열이다.($h\ \times\ w\ \times\ d$)
- 첫 번째 layer는 pixel size($h\ \times\ w$), 그리고 $d$ channels이다.
- 상위 layer의 위치는 해당 layer가 경로로 연결된 이미지의 위치에 해당하며, 이를 $receptive fields$라고 한다.
- 기본 구성 요소(convolution, pooling 및 activation function)는 로컬 입력 영역에서 작동하며 상대적인 공간 좌표에만 의존한다.
 - $x_{ij}$ : data vector at location (i,j)

 - $y_{ij}$ : following layer

 - $k$ : kernel size

 - $s$ : stride or subsampling factor
 
 - $f_{ks}$ : determines the layer type <br>
 (e.g. multiplication for convolution or average pooling,<br>
a spatial max for max pooling, an elementwise,<br>
nonlinearity for an activation function, etc.)

$$y_{ig}\ =\ f_{ks}(\ \{ x_{si+\delta i,\ sj+\delta j} \}_{0\leq\delta i,\ \delta j < k}\ )$$

  

<img src = 'https://drive.google.com/uc?id=1ThH_z5nN_v_8XABLp-aBFJOnZIHq1Rg1' height = 500 width = 700>

위의 함수의 형태는 transformation rule에 따라 kernel size, stride를가진 구성하에 유지된다.

$$f_{ks}\ \circ\ g_{k^{'}s^{'}}\ =\ (f\ \circ\ g)_{k^{'}\ +\ (k-1)s^{'},\ ss{'}}$$

일반적인 net은 일반적인 nonlinear function을 계산하지만, 이러한 형태의 layers는 nonlinear filter만을 계산한다.
<br>
---
<br><br>

<span style="color:red"><bold>Loss Function</bold></span>
- FCN으로 구성된 loss function은 task를 정의한다.
- 만약 loss function이 final layer의 공간 차원의 합이라면($l(x;\theta\ =\ \sum ;^{'}(x_{ij};\theta)$ ),loss function의 parameter gradient는 각 공간의 parameter gradients의 합이 된다.
 * (전체 이미지를 계산한 $l$에서 통계적 gradient descent) = (모든 final layer receptive fields를 minibatch로 가져오는$l^{'}$에서의 통계적 gradient descent)

## A. Adapting classifiers for dense prediction

-  LeNet, AlexNet 등 전형적인 인식 nets에서 <span style = "color:blue"><bold>fully connected layers</bold></span>는 고정된 차원을 가지고, 공간좌표를 무시한다.
 * <span style = "color:blue"><bold>fully connected layers</bold></span>는 entire input regions을 덮는 kernel을 가진 convolution으로 볼 수도 있다.
- 그렇게 하면 이러한 net가 모든 크기의 입력을 받고 **spatial output maps**을 만드는 <span style = "color:green"><bold>fully convolutional networks</bold></span>로 캐스팅된다.
 * spatial output maps는 semantic segmentation과 같은 고밀도 문제를 자연스러운 선택을 하도록한다.

<img src = 'https://drive.google.com/uc?id=12Nltu7taZSLlSyj3a_j7lbDZnk6sFiBp' height = 500 width = 700>

VGG16을 예로 살펴보았을 때, 다음과 같이 출력층 부분의 마지막 3 fc(fully connectd) layers를 모두 Conv-layers로 변경한다.

<img src = 'https://drive.google.com/uc?id=1875Kubh_pKBFKsX0fO1ZG12-yUMivEFk' height = 500 width = 700>

<span style = "color:blue"><bold>Dense Layer에서 Conv Layer로 변환하는 방식은 다음과 같다.</bold></span>

<img src = 'https://drive.google.com/uc?id=1EEJhmEnwGbLk7kRW7eVwWyFgILd7vrq3' height = 500 width = 700>

 첫 번째 fc layer를 $(7 \times 7 \times 512)$ 4096 filter conv로 변경하면 가중치의 수가 유지된다.

<img src = 'https://drive.google.com/uc?id=1yCiRnh4QommAZeA7OljY6fzyMRo1Oep3' height = 500 width = 700>

마지막 fc-layer의 경우 채널 차원을 클래스 수에 맞춤 $1 \times 1 $ conv로 변환한다

- fully convolutional에서의 classification nets의 재해석은 input size에 상관없이 output maps를 생성하지만, 그 output 차원은 subsampling에 의해 줄어든다.
- classification nets는 필터를 작고 계산 요구 사항을 합리적으로 유지하기 위해 subsampling한다.
- 이것은 이러한 nets의 fully convolutional version의 출력을 강화하여 출력 장치의 receptive fields의 pixel stride와 동일한 인수로 입력 크기부터 줄인다.

---

Convolutionalization을 통해 출력 Feature map은 원본 이미지의 위치 정보를 내포할 수 있게 되었다.

그러나 Semantic segmentation의 최종 목적인 픽셀단위 예측과 비교했을 때, FCN의 출력 Feature map은 너무 coarse하다.

따라서 Coarse map을 원본 이미지 크기에 가까운 Dense map으로 변환해야 하는데, 적어도  $input\ image\ size\ \times\ \frac{1}{32}$보다는 해상도가 높을 필요가 있다.
<br>$\Rightarrow$ Deconvolution 필요하다.
> Coarse map에서 Dense map을 얻는 몇 가지 방법이 존재한다.
> * Unpooling
>>이 경우 필터가 더 세밀한 부분은 볼 수 있지만, Receptive Field가 줄어들어 이미지의 context를 놓치게 된다. 또한 학습 파라미터 수가 급격히 증가하고 더 많은 학습시간을 요구하게 된다.
> * Upsampling
> * Shift and stitch

## B. Shift-and-stitch is filter dilation
- shifted versions of the input의 outputs을 stitching하여 coarse outputs에서 dense prediction을 얻을 수 있다.
- outputs이 $f$ 계수로 downsampling되는 경우 input $x$ 픽셀을 **오른쪽**으로, $y$ 픽셀을 $0 ≤ x, y < f$와 같이 모든 $(x, y)$에 대해 한 번씩 **아래**로 이동시킨다.

<img src = 'https://drive.google.com/uc?id=1sU4fLPlZIPhps1xMAJwX6wfwXKvFKg3c' height = 500 width = 700>

- 빨간색 박스는 2 X 2 크기의 maxpooling filter로 stride는 2로 오른쪽 3 X 3크기의 행렬을 보여준다.
- 나머지 색 박스의 연산은 위의 빨간색 박스와 동일한 연산을 하지만 입력행렬이 다르다.
- 오른쪽 3X3 행렬 중에 회색으로 표시되어 있는 부분은 입력 이미지를 1 pixel만큼 이동했기 때문에 필요없는 정보가 된다.
- 위의 과정을 pixel 이동 방향을 다르게 하여 여러 번 반복한다.
- 어디서 max pooling 연산이 적용되었는지에 대한 위치 정보만 저장한다면 3 X 3 image를 5 X 5 image로 upsamling 가능하다.
- 이 알고리즘의 치명적 단점은 계산 비용이 크다.

본 논문에서는dilation에 대한 예비 실험을 했지만, 모델에는 사용하지 않았다. 

다음 절에서 Upsampling을 통한 학습이 특히 나중에 설명하는 skip layer fusion과 결합될 때 효과적이고 효율적이라는 것을 발견하기 때문이다..

## C. Upsampling is (fractionally sstrided) convolution

본 논문에서는 Bilinear Interpolation과 Backwards convolution을 사용하였다.

### **Bilinear Interpolation**



<img src = 'https://drive.google.com/uc?id=1AWKdZjeBCuE2k6zWzDanuKP2RHNb47A7' height = 500 width = 700>

simple bilinear interpolation은 input 및 output 셀의 상대적인 위치에만 의존하는 선형 맵에 의해 가장 가까운 4개의 입력으로부터 각 출력 $y_{ij}$를 계산한다.

$$
y_{ij}\ =\ \sum_{\alpha ,\beta = 0}^{1}|1\ -\ \alpha\ -\ \{i/f\}|\ |1\ -\ \beta\ -\ \{i/j\}|\ x_{\lfloor i/f \rfloor\ +\ \alpha,\ \lfloor j/f \rfloor\ +\ \beta}\\ 
f\ :\ upsampling\ factor
$$

### **Backwards convolution**

- 어떤 의미에서, factor $f$ upsampling은 input stride가 $\frac{1}{f}$인 convolution이다.

- $f$가 적분되어 있을 때, 전형적인 input-strided convolution의 forward, backward를 역전시킴으로써 "backward convolution을 통해 upsampling을 구현하는 것이 자연스럽다.
 - Thus upsampling is performed in-network for end-to-end learning by back-propagation from the pixelwise loss

<img src = 'https://drive.google.com/uc?id=1qfLWVO_NJq1cZCM5LgQfhWCN34u8s4d4' height = 500 width = 700>

**Deconvolution**

- 이러한 계층의 convolution filter는 고정될 필요가 없지만(예를 들어, bilinear upsampling을 위해) 학습할 수 있다.

- Deconvolution layers 및 activation function 의stack은 nonlinear upsampling을 학습할 수도 있다.

- 본 논문의 실험에서, 네트워크 내 upsampling이 dense prediction을 학습하는 데 빠르고 효과적이라는 것을 발견했다.

## D. Patchwise training is loss sampling

- patchwise training이란 모델을 training 할 때 전체 이미지를 넣는 것이 아니라 객체의 주변을 분리한 서브 이미지 즉 patch image를 사용하는 방법을 의미한다. 
- 여기서 이미지의 부분을 학습하는 patch training과 전체 이미지를 학습하는 fully convolution training은 서로 동일하다. 
- 본 논문에서는 fully convolution training이 더 계산적으로 유리하다고 언급한다.

# **III. Segmentation**

- 본 논문에서는 ILSVRC classifiers를 FCNs으로 변경시키고, upsampling and a pixelwise loss를 가진 dense prediction을 위해 증강시켰다. 
 - ILSVRC2012 : AlexNet
 - ILSVRC2014 : VGG16
 - ILSVRC2014 : GoogLeNet
- fine-tuning에 의해 segmentation을 train 한다.
 - <span style = "color:blue"><bold>fine-tuning : 기존에 학습된 모델을 기반으로 새로운 목적에 맞게 변형하여, 학습된 모델의 가중치를 미세하게 조정하여 재학습 하는 방법이다.</bold></span>
- **skip connection**을 coarse, semantic한 layer와 local,appearance한 layer 사이에 추가한다.
- skip connection을 추가한 네트워크 구조를 skip architecture라고 하는데, skip architecture는 end-to-end로 학습된다.
- 본 논문에서 사용된 데이터 셋은 PASCAL VOC 2011 segmentation challenge에서 사용된 데이터이다.
- 학습을 할 때는 pixel 별 multinomial logistic loss를 사용했습니다. multinomial logistic loss은 아래와 같다.

$$
J(\theta)\ =\ -\frac{1}{m}\ [\ \sum_{i=1}^{m}\ y^{(i)}\ log\ h_{\theta}(x^{(i)})\ +\ (1\ -\ y^{(i)})\ log(1\ -\ h_{\theta}(x^{(i)}))] \\ 
y\ :\ true\ label\\ 
h_{\theta}\ :\ model\\ 
x\ :\ input\ data
$$
- 마지막 성능 평가를 위해 IoU를 사용하였다.


<img src = 'https://drive.google.com/uc?id=1N26a2qWHH-vH2Ap93oE74fs5uqUt-vHC' height = 500 width = 700>

## A. From classifier to dense FCN

- GoogLeNet에서 final loss layer만을 사용했고, final average pooling layer을 통해 수행을 개선시켰다.
- 각 net에서 final classifier layer을 제거하고 fully connected layers를 fully convolutions으로 변경했다.
- channel이 21인 1 X 1 convolution을 추가하여 각 coarse output locations에서 PASCAL class(배경포함)에 대한 점수를 예측한다. 이후 pixelwise outputs에 대한 coarse outputs을 **bilinearly upsampling**하기 위해 뒤로 convolution layer를 추가한다.
- Train by SGD with momentum. Gradients are accumulated over 20 images.
- Set fixed learning rates of 10−3, 10−4, and 5−5 for FCN-AlexNet, FCN-VGG16, and FCN-GoogLeNet
- Use momentum 0.9, weight decay of 5−4 or 2−4, and doubled learning rate for biases.
- 무작위 초기화가 더 나은 성능이나 더 빠른 수렴을 제공하지 않기 때문에 클래스 점수 매기기 계층을 0으로 초기화한다.
- 드롭아웃은 original classifier net에서 사용되는 곳에 포함된다.

## B. Image-to-image learning

- 본 논문에서는 every pixel이 batch, image dimensions와 관계없이 같은 가중치를 가지도록 하기 위해 loss normalize를 하지 않는다.
 - 따라서 loss는 all pixels의 합이기 때문에 small learning rate를 사용한다.

## C. Combining what and where

FCN base networks scoe는 높은 성과를 거뒀지만 불만족스러운 coarse이다.

stride of the network prediction은 upsampling된 결과에서 detail을 제한한다.
 > $\Rightarrow$ 우리는 특히 예측에서 더 미세한 진보를 보이는 더 얕은 레이어를 포함하기 위해 layer outputs을 융합하는 skip을 추가하여 이를 해결한다.

**[Skip connection Architecture]**

<img src = 'https://drive.google.com/uc?id=13fG1GN-ESi2HMOBbWcd48saFP9mKmKO0' height = 500 width = 700>

- Skips으로 보강된 network는 함꼐 학습된 several streams에서 prediction을 만들고 통합한다.
- fine layers와 coarse layres를 결합하면 전역구조를 무시하지 않고 local predictions이 가능하게 한다.
- Scale agreement
 - 네트워크 내에서 저해상도 layer를 upsampling하여 두 layers를 scale agreement로 가져온다.
 - Cropping을 가지고 padding으로 인해 다른 layers를 넘어 확장된 upsampled layer의 모든 부분을 제거한다.
 > $\Rightarrow$ : 정확하게 동일한 치수의 layer를 만든다.
- Fuses predictions
 - 층을 공간적으로 정렬한 다음 융합 연산을한다.
 > $Rightarrow$ : 연결을 통해 features를 융합하고, 즉시 1 X 1 Conv로 구성된 "Score Layer"에 의한 분류를 따른다.

$\bf{\color{red} {\Rightarrow}}$  <span style = "color:red"><bold>모든 layers가 융합되면, 최종 예측이 영상 해상도로 다시 업샘플링된다.</bold></span>
