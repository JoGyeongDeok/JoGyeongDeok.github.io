---
title: A competition to optimize the environment for growth3
tags: [Dacon, Deep Learning, Computer Vision]
excerpt: Multi-Modal

date: 2022-5-8
categories: 
  - Project
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Multi-Modal
---

<STYLE TYPE="text/css">
    table {font-size: 13pt;}
</STYLE>

# <a href= "https://dacon.io/en/competitions/official/235897/overview/description">생육 환경 최적화 경진대회</a>

<img src = "/images/optim/optim_1.jpg" height = "700" width = "700">


|   |   |
|:-:|:---|
|**수행 기간**|2022.04 - 2022.5|
|**과제 개요**|청경채 사진과 환경데이터를 활용해 잎면적 예측|
|**참여 인원**|2|
|**사용 언어**|Python, Pytorch|
|**담당 역할**|-이미지 Agumentation <br>- 이미지 데이터와 환경 데이터로 하나의 신경망 구축|
|**주요 기능**|- 이미지 데이터, 환경 데이터를 활용해 잎면적 예측|
|**성과**| - 총 136팀 중 리더보드 9위<br> - 최종 탈락|



<br>

# 1. Resnet 모델을 통해 Image Feature Extractor 

<img src = "/images/optim/optim_3.jpg" height = "700" width = "700">

이미지 데이터와 환경데이터를 동일한 신경망에 구축하기 위해 Resnet 모델을 사용하여 Image Feature Extractor 구축하였다.

<br>

# 2. 환경 데이터를 통해 MetaBlock Layer 생성 

<img src = "/images/optim/optim_2.jpg" height = "700" width = "700">

환경데이터를 통해 MetaBlock Layer를 생성한 후 CNN모델을 통해 추출한 Image Feature와 MetaBlock을 결합하여 Regression 결과를 얻을 수 있었다.


# <a href= "https://github.com/JoGyeongDeok/Project/blob/main/Dacon/2022_05_08_A_competition_to_optimize_the_environment_for_growth3.ipynb">실행 코드</a>