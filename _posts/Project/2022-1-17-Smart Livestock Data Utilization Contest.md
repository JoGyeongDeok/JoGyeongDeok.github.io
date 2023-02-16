---
title: Smart Livestock Data Utilization Contest
tags: [Ai Factory, Deep Learning, Computer Vision]
image : http://cdn.aifactory.space/images/20211220222057_lvJw.jpg
excerpt: Semantic Segmentation
date: 2022-1-17
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

# <a href= "https://aifactory.space/competition/detail/1952">스마트축사 데이터 활용 대회</a>



<img src = "/images/LiveStock/LiveStock_1.jpeg" height = "700" width = "700">


|   |   |
|:-:|:---|
|**수행 기간**|2022.01 - 2022.02|
|**과제 개요**|스마트 축사 데이터를 활용하여 한우의 발정행동을 판별하는 모델 개발|
|**참여 인원**|1|
|**사용 언어**|Python, Pytorch|
|**담당 역할**|- Annotation File 및 이미지 처리<br>- MaskRCNN 모델링<br>- 학습 및 예측|
|**주요 기능**|- CCTV 사진을 통해 한우 Segementation<br>- 한우의 발정여부 Classification|
|**성과**| - 리더보드 3위<br> - 장려상 수상|

<br>

# 1. MaskRCNN을 통해 모델 튜닝 학습

<img src = "/images/LiveStock/LiveStock_2.png" height = "700" width = "700">

이미지 데이터 Agumentation이후 Segmentation을 위해 MaskRCNN을 사용하였다.


<br>

# 2. 학습된 모델을 통해 Inference 시행

<img src = "/images/LiveStock/LiveStock_3.png" height = "700" width = "700">

학습된 모델을 통해 CCTV 이미지를 Segmentation 및 Classification을 수행하였다.