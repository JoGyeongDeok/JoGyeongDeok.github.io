---
title: Logistics distribution forecasting contest
tags: [Dacon, Machine Learning, Data Analysis]
image : 
excerpt: Time Series
date: 2021-12-20
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

# <a href= "https://dacon.io/competitions/official/235867/overview/description">물류 유통량 예측 경진대회</a>

<img src = "/images/Logistics/Logistics_1.jpeg" height = "700" width = "700">



|   |   |
|:-:|:---|
|**수행 기간**|2021.12|
|**과제 개요**|제주시 택배 운송 데이터를 이용하여 운송장 건수 예측|
|**참여 인원**|2|
|**사용 언어**|Python|
|**담당 역할**|- 데이터 EDA 및 전처리<br>- 파생변수 생성<br>- 모델 튜닝 및 예측|
|**주요 기능**|- 송하인, 수하인, 물품 정보를 활용하여 운송장 건수 예측|
|**성과**| - 총 237팀 중 리더보드 1위<br> - 1등 수상|

<br>

# 1. 데이터 전처리 및 파생변수 생성

입력 데이터로 송하인_격자공간고유번호, 수하인_격자공간고유번호, 물품_카테고리가 존재하였고, Target 값으로 운송장_건수가 존재하였다. 
데이터를 탐색한 결과 **격자공간고유번호**가 1~5, 6~9, 10, 11~16 자리로 구분지어 의미가 있다고 판단하였다. 이를 통해 격자공간고유번호 구간별 파생변수를 생성하였다.

<br>

# 2. 모델링

입력 데이터 모두 범주형 데이터이므로 **CatBoost** 모델을 사용하는 것이 좋다 판단했고, **Optuna**를 이용해 하이퍼파라미터를 튜닝하였다. 또한 **Stratified 10-fold**를 활용하여 더욱 **Robust**한 모델을 생성하여 최종결과를 예측하였다.