---
title: The 7th Lotte Members Big Data Contest
tags: [Ai Factory, Machine Learning, Data Analysis]
excerpt: Data Analysis
date: 2022-8-31
categories: 
  - Project
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Data Analysis
---

<STYLE TYPE="text/css">
    table {font-size: 13pt;}
</STYLE>


# <a href= "https://aifactory.space/competition/detail/2063">제7회 롯데멤버스 빅데이터 경진대회</a>

<img src = "/images/lotte/lotte_1.jpeg" height = "650" width = "650">


|   |   |
|:-:|:---|
|**수행 기간**|2022.07 - 2022.8|
|**과제 개요**|고객 구매 데이터를 통해 예측 모델 개발 및 비즈니스 아이디어 제안|
|**참여 인원**|3|
|**사용 언어**|Python, Pytorch|
|**담당 역할**|- 데이터 전처리 및 상관분석<br>- 예측 모델링|
|**주요 기능**|- RNN 오토인코더를 활용해 고객의 영수증 군집화<br>- 군집화된 영수증을 통해 확률분포 간 유사도 추정<br>- 최종적으로 구매 이력 유사도가 높은 다른 고객 추천|
|**성과**| - 최종 탈락|

<br>

# 1. EDA

<img src = "/images/lotte/lotte_3.png" height = "650" width = "650">

EDA를 통해 연령대 별 집단의 특징을 분석했다. 20대는 잠재고객, 30, 40대는 정착기 고객, 50대 이상은 공략하기 어려운 연령층으로 해석할 수 있었다. **이용량 비율 차**와 **사용자 비율 차**를 비교하여 연령대별 특징을 분석하였다. 

<br>

# 2. 고객 추천 시스템 모델(LSTM AutoEncoder)

<img src = "/images/lotte/lotte_5.png" height = "650" width = "650">

LSTM AutoEncoder 모델를 이용해 **Embedding** 벡터를 추출하여 고객들을 묶는 **Fuzzy Clustering**을 진행하였다. 이 후 고객과 고객을 추천하는 시스템을 기획했으며, 시뮬레이션 결과 랜덤하게 상품을 추천한 것보다 **7배** 높은 Score를 얻을 수 있었다.

<br>

# 2. 영수증 추천 마케팅

<img src = "/images/lotte/lotte_6.png" height = "650" width = "650">

영수증을 통해 구매정보를 공유하는 고객들은 개인의 **개성들을 표현**할 수 있고, 영수증을 추천받은 고객들에게는 하나의 컨텐츠가 될 수도 있으며 고객들의 **구매심리**를 자극할 수 있다.  