---
title: AI SPARK Challenge of the 2nd R and D Special Zone Artificial Intelligence Competition
tags: [Ai Factory, Deep Learning, Computer Vision]
excerpt: Object Detection
date: 2022-2-4
categories: 
  - Project
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Compter Vision
---


<STYLE TYPE="text/css">
    table {font-size: 13pt;}
</STYLE>

# <a href= "https://aifactory.space/competition/detail/1946">제2회 연구개발특구 인공지능 경진대회 AI SPARK 챌린지</a>

<img src = "/images/spark/spark_1.jpeg" height = "700" width = "700">


|   |   |
|:-:|:---|
|**수행 기간**|2022.01 - 2022.02|
|**과제 개요**|CCTV이미지를 통해 가축 Object Detection|
|**참여 인원**|2|
|**사용 언어**|Python, Pytorch|
|**담당 역할**|- Annotation File 및 이미지 처리<br>- Yolo 모델을 사용해 모델 개발 |
|**주요 기능**|- 가중치 크기 100MB 이하 조건에서 Object Detection 지표 mAP 최적화|
|**성과**| - 리더보드 11위<br> - 최종 탈락|



<br>

# 1. Yolo를 통해 모델 튜닝 학습

<img src = "/images/spark/spark_2.jpg" height = "700" width = "700">

이미지 데이터 Agumentation이후 Object Detection을 실시하였다. 또한 가중치 100MB이하 조건이 있기 때문에 Real Time Detection이 가능한 Yolo 모델을 사용하였다.


<br>

# 2. 학습된 모델을 통해 Inference 시행

<img src = "/images/spark/spark_3.png" height = "700" width = "700">

학습된 모델을 통해 CCTV 이미지에서 가축을 Detection하였다.

# <a href= "https://github.com/JoGyeongDeok/Project/blob/main/AIFactory/2022_2_4_AI_SPARK_Challenge_of_the_2nd_R%26D_Special_Zone_Artificial_Intelligence_Competition.ipynb">실행 코드</a>