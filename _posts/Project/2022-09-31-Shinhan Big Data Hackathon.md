---
title: The 1st Shinhan Big Data Hackathon
tools: [Machine Learning, Data Analysis ]
excerpt: Data Analysis
date: 2022-9-31
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


# 신한 빅데이터 해커톤

<img src = "/images/sinhan/sinhan_1.png" height = "650" width = "650">


|   |   |
|:-:|:---|
|**수행 기간**|2022.09|
|**과제 개요**|신한카드 고객 이용내역 데이터를 활용한 신한은행의 비즈니스 모델 제안 |
|**참여 인원**|5|
|**사용 언어**|Python|
|**담당 역할**|- 데이터 전처리<br>- 주성분 분석, 군집 분석|
|**주요 기능**|- 주성분 분석 후 해석 가능한 군집화 실시<br>- 해석한 군집을 바탕으로 비즈니스 모델 제안|
|**성과**| - 최종 우수상

<br>

# 1. 카드 이용 내역 데이터를 통한 패턴 분석(PCA)

<img src = "/images/sinhan/sinhan_2.png" height = "650" width = "650">

주성분 분석을 통해 카드 이용 패턴 분석을 하였다. 주성분 분석에서 **각 주성분이 이용패턴이 되며** 주성분의 성분에 대한 **방향**과 비중을 통해 이용패턴의 이름을 결정하였다. 

<br>

# 2. 이용패턴을 통한 군집분석(군집분석)

<img src = "/images/sinhan/sinhan_3.png" height = "650" width = "650">
<img src = "/images/sinhan/sinhan_4.png" height = "650" width = "650">

각 이용패턴을 통해 **군집화**를 실시하였고 **군집별 은행활동 비율**을 구하였다. 군집별 이용패턴과 은행활동 비율이 설명이 되는 것을 보아 알맞게 군집분석이 이루어 졌다고 볼 수 있다.

<br>

# 3. 고객군 특성별 상품 추천(비즈니스 모델 제안)

<img src = "/images/sinhan/sinhan_5.png" height = "650" width = "650">
<img src = "/images/sinhan/sinhan_6.png" height = "650" width = "650">

군집별 은행활동 유무, 소비패턴 등을 통해 설명 가능한 신용도를 결정하였고, 이를 통해 비즈니스 모델을 제안하였다.