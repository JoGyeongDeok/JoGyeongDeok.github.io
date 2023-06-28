---
title: "Linear Algebra - 4(Determinants)"
category: Mathematics
tags: [Linear Algebra, Mathematics, Determinants]
comments: true
date : 2023-05-11
categories: 
  - blog
excerpt: Contemporary Linear Algebra
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---
<br>

# 1. Determinants; Cofactor Expansion

**Minors and Cofactors**: 만약 $A$가 정방행렬이라면
- $M_{ij}$를 **minor of entry $a_{ij}$**
- Cofactor는 $C_{ij}=(-1)^{i+j}M_{ij}$

$$M_{11}=\begin{vmatrix}\color{}3 & 1 &-4\\2&\color{red}5&\color{red}6\\1&\color{red}4&\color{red}8 \end{vmatrix} = \begin{vmatrix}5&6\\4&8\end{vmatrix}\ =\ 16,\ \ \ C_{11} = (-1)^{1+1}M_{11}=16$$<br><br>
$$M_{32}=\begin{vmatrix}\color{red}3 & 1 &\color{red}{-4}\\\color{red}2&5&\color{red}6\\1&4&8 \end{vmatrix} = \begin{vmatrix}3&-4\\2&6\end{vmatrix}\ =\ 26,\ \ \ C_{32} = (-1)^{3+2}M_{32}=-26$$

<br>
만약 행렬 A가 $n\times n$정방행렬이고, $1\leq i \leq n,\ \ 1\leq j \leq n$ 일 때,
- $det(A)\ =\ \sum_{i}^{n}a_{ij}C_{ij}$
- $det(A)\ =\ \sum_{j}^{n}a_{ij}C_{ij}$

$$A = \begin{bmatrix}2&0&0&5\\-1&2&4&1\\3&0&0&3\\8&6&0&0\end{bmatrix},\\ \\ \\ \\ \\  det(A)\ =\ (0)C_{13} + (4)C_{23}+(0)C_{33}(0)C_{43}\\ =(4)C_{23}\\
=(-4) \begin{vmatrix}2&0&5\\3&0&3\\8&6&0\end{vmatrix}\\ = -216$$

<br><br>

# 2. Propertie of Determinants

만약 $A$가 정방행렬이라면, 
- $det(A) = det(A^T)$
- 만약 행렬 B가 A의 single column or single row가 상수 K배 곱해진 결과라면, $det(B)=kdet(A)$
  - $$B = \begin{vmatrix}ka_{11}&ka_{12}&ka_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{vmatrix}$$
- 만약 행렬 B가 A의 two column or two row가 interchanged 된 결과라면, $det(B)=-det(A)$
  - $$B = \begin{vmatrix}a_{21}&a_{22}&a_{23}\\a_{11}&a_{12}&a_{13}\\a_{31}&a_{32}&a_{33}\end{vmatrix}$$
- 만약 행렬 B가 행렬A의 배수만큼 더해진 one row or column의 결과라면, $det(B)=det(A)$ 
  -  $$B = \begin{vmatrix}a_{11}+ka_{21}&a_{12}+ka_{22}&a_{13}+ka_{23}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{vmatrix}$$
- 만약 A가 두 개의 **identical** rows 혹은 columns을 가진다면, $det(A) = 0$
- 만약 A가 두 개의 **proportional** rows 혹은 columns를 가진다면, $det(A) = 0$
- $det(kA)\ =\ k^ndet(A)$