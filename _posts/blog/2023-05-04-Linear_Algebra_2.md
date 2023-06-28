---
title: "Linear Algebra - 2(Systems of Linear Equations)"
category: Mathematics
tags: [Linear Algebra, Mathematics]
comments: true
date : 2023-05-04
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

# 1. Introduction to Systems of Linear Equations

<br>
**Linear Equation** : $a_1x_1+a_2x_2+...+a_nx_n=b$<br>
**Homogeneous Linear Equation** : $a_1x_1+a_2x_2+...+a_nx_n=0$<br>
**Trivial Solution** : $A\bf x = 0$일 때 $\bf x = 0$이 유일한 해이다.
<br>

**Augmented Matrices and Elementary row operations**<br>

$$a_{11}x_1 + a_{12}x_2 + ... +a_{1n}x_n = b_1 \\a_{21}x_1 + a_{22}x_2 + ... +a_{2n}x_n = b_2\\ \underset{.}{\underset{.}{.}} \\ a_{m1}x_1 + a_{m2}x_2 + ... +a_{mn}x_n = b_m$$

위와 같은 Linear Equation들을 아래의 행렬처럼 표현할 수 있다. 이 때 아래의 행렬을 **augmented matrix**라 부른다.<br>

$$\begin{bmatrix}a_{11} & a_{12} &... &a_{1n} &b_{1} \\a_{11} & a_{12} &... &a_{1n} &b_{1} \\&& \underset{.}{\underset{.}{.}} \\ a_{11} & a_{12} &... &a_{1n} &b_{1} \end{bmatrix}$$

<br><br>

# 2. Solving Linear Systems by Row Reduction

<img src = '/images/Linear Algebra/1_4.jpg' height = 300 width = 600>


## **echelon form**<br>
- 모든 nonzero 행은 zero행 위에 존재해야 한다.
- 각 nonzero 행의 leading entry(맨 처음 원소)는 윗(nonzero)행의 leading entry보다 오른쪽에 있어야 한다.
- 각 열에서 leading entry 아래의 항목들은 모두 0이어야 한다.
<br>

## **Row Reduced Echelon Form(RREF)**<br>
- Echelon Form이어야한다.
- leading entry는 모두 1이어야 한다.
- 각 열에서 leading entry 1를 제외하고 나머지 항목은 모두 0이다.

## **elementary row operation**<br>
- Interchange two rows
- Multiply a row by a nonzero constant
- Add a multiple of one row to another


<br>

## **Gauss-Jordan and Gaussian Elimination**<br>


$$\begin{bmatrix}0 & 0&-2&0&7&12 \\2&4&-10&6&12&28\\2&4&-5&6&-5&-1 \end{bmatrix}$$

위와 같은 **augmented matrix**를 Gauss-Jordan and Gaussian Elimination를 통해 해결해보겠다.<br>

***Step 1.*** : 전체가 0으로 이뤄지지 않은 열을 맨 왼쪽에 위치시킨다.

$$\begin{bmatrix}0 & 0&-2&0&7&12 \\2&4&-10&6&12&28\\2&4&-5&6&-5&-1 \end{bmatrix}$$ 변경할 필요가 없다.

***Step 2.*** : 첫 번째 항목에 0이 오지 않도록 top row와 또 다른 row의 위치를 서로 바꾼다. 

$$\begin{bmatrix}2&4&-10&6&12&28\\0 & 0&-2&0&7&12 \\2&4&-5&6&-5&-1 \end{bmatrix}$$ 첫 번째 행과 두 번째 행의 위치를 서로 바꿨다.


***Step 3.*** : Step1에서 찾은 열의 가장 위의 row의 항목이 $a$라면 그 row에 $1/a$를 곱하여 1로 만들어준다. 

$$\begin{bmatrix}1&2&-5&3&6&14\\0 & 0&-2&0&7&12 \\2&4&-5&6&-5&-1 \end{bmatrix}$$ 첫 번째 행에 $\frac12$를 곱하였다.


***Step 4.*** : 맨 위 행에 적합한 배수를 아래 행에 더하여 맨 위 행의 1 아래의 모든 항목이 0이 되도록 합니다. 

$$\begin{bmatrix}1&2&-5&3&6&14\\0 & 0&-2&0&7&12 \\0&0&5&0&-17&-29 \end{bmatrix}$$ 첫 번째 행의 -2배를 마지막 행에 더하였다.

***Step 5.*** : 이제 맨 위의 행은 내비두고, 아래 하위 행렬에 대해 Step1~4를 반복하여 최종적으로 전체 행렬이 **row echelon form**이 되도록 한다.

$$\begin{bmatrix}1&2&-5&3&6&14\\0&0&1&0&-\frac72&-6 \\0&0&5&0&-17&-29 \end{bmatrix}$$ 두 번째 행에 $-\frac12$을 곱하였다.

$$\begin{bmatrix}1&2&-5&3&6&14\\0&0&1&0&-\frac72&-6 \\0&0&0&0&\frac12&1 \end{bmatrix}$$ 두 번째 행의 -5배를 마지막 행에 더하였다.

$$\begin{bmatrix}1&2&-5&3&6&14\\0&0&1&0&-\frac72&-6 \\0&0&0&0&1&2 \end{bmatrix}$$ 마지막 행에 2를 곱하였다.

전체 행렬이 **row echelon form**형태가 되었다.


***Step 6.*** : 마지막 0이 아닌 행부터 시작하여 위쪽으로 작업하고 위의 행에 각 행의 적절한 배수를 더하여 각 행의 선두 1 위에 0을 추가합니다.

$$\begin{bmatrix}1&2&-5&3&6&14\\0&0&1&0&0&1 \\0&0&0&0&1&2 \end{bmatrix}$$ 마지막 행에 $\frac72$를 곱하여 두번 째 행에 더하였다.

$$\begin{bmatrix}1&2&-5&3&0&2\\0&0&1&0&0&1 \\0&0&0&0&1&2 \end{bmatrix}$$ 마지막 행에 $-6$를 곱하여 첫번 째 행에 더하였다.

$$\begin{bmatrix}1&2&0&3&0&7\\0&0&1&0&0&1 \\0&0&0&0&1&2 \end{bmatrix}$$ 두 번째 행에 $5$를 곱하여 첫번 째 행에 더하였다.

마지막 결과인 

$$\begin{bmatrix}1&2&0&3&0&7\\0&0&1&0&0&1 \\0&0&0&0&1&2 \end{bmatrix}$$

행렬은 **reduced row echelon form(RREF)**이다. 이러한 절차를 ***Gauss-Jordan elimination***이라 한다.