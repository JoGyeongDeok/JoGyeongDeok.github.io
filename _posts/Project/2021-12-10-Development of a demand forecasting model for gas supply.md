---
title: Development of a demand forecasting model for gas supply
tags: [Dacon, Machine Learning, Time series]
excerpt: Time Series
date: 2021-12-10
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

# <a href= "https://dacon.io/competitions/official/235830/overview/description">가스공급량 수요예측 모델개발</a>

<img src = "/images/GAS/GAS_1.jpeg" height = "600" width = "600">



|   |   |
|:-:|:---|
|**수행 기간**|2021.11 - 2021.12|
|**과제 개요**|시간단위 가스공급량 예측 모델 개발|
|**참여 인원**|5|
|**사용 언어**|Python|
|**담당 역할**|- 데이터 EDA 및 전처리<br>- 외부데이터 수집 및 파생변수 생성<br>- 모델 튜닝 및 예측|
|**주요 기능**|- 시간 별 가스 공급량 데이터와 기상정보 외부 데이터를 활용하여 90일까지의<br> 시간 당 가스 공급량 예측|
|**성과**| - 총 68팀 중 리더보드 9위(규정 위반 팀 제외)<br> - 특별상 수상|

<br>

# 1. 데이터 전처리

<img src = "/images/GAS/GAS_2.jpeg" height = "600" width = "600">

각 구분 별 BoxPlot을 보면 이상치가 많이 존재한다. 모델링 전에 이상치를 처리해야 한다고 생각하여 4-sigma를 벗어나는 값을 이상치로 판단하여 제거하였다.

<br>

# 2. 파생변수 생성

<img src = "/images/GAS/GAS_4.jpeg" height = "600" width = "600">

연월일을 통해 Month, Year, Weekday와 Holiday 파생변수들을 생성하였다. 또한 시간 당 공급량을 통해 **일간 공급량**, **하루 가스 공급량 시간당 비율** 파생변수를 생성하였다.

<br>

# 3. 최종 예측

<img src = "/images/GAS/GAS_4.jpeg" height = "600" width = "600">

시계열 모델의 경우 시간 당 예측보다 일간 예측이 더 **Robust** 하고, **하루 가스 공급량 시간당 비율**도 비교적 **Robust** 하다고 생각하여 일간 예측과 시간당 비율을 각각 모델링 및 예측한 결과를 곱하여 최종 결과물로 제출하였다.