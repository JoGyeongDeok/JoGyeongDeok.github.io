---
title: "Linear Algebra - 3(Matrices and Matrix Algebra)"
category: Mathematics
tags: [Linear Algebra, Mathematics, LU-Decomposition]
comments: true
date : 2023-05-05
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

# 1. Operations on Matrices

- $$A{\bf x}=[{\bf a_1 a_2 ... a_n}] \begin{bmatrix} x_1 \\ x_2 \\ ... \\ x_n \end{bmatrix}=x_1{\bf a_1} + x_2{\bf a_2} + ... + x_n{\bf a_n}$$.
- $$AB=[A{\bf b_1} A{\bf b_2} ... A{\bf b_n}]$$.
- $A(c{\bf u})$
- $A({\bf u + v})= A{\bf u}+ A{\bf v}$
- Transpose of A Matrix : $A_{ij}=(A^T)_{ji}$ 
- 만약 A가 정방행렬이라면 trace가 존재한다. 이 때 trace는 행렬 A의 대각원소 합이다. : $tr(A)$
- Inner Product : $\bf u^Tv$
- Outer Product : $\bf uv^T$

<br>
<br>

# 2. Inverses; Algebraic Properties of Matrices

- **Zero Matrices** : 모든 원소가 0인 행렬
- **Identity Matrices($I_n$)** : 대각원소가 모두 1 이고 비대각원소가 모두 0인 행렬
  - $AI_n=A$ and $I_mA=A$
  - 행렬 $R$이 정방행렬 $A$의 **the reduced row echelon form**이라면, R은 zero 행을 가지거나 Identity Matrix $I_n$ 이다. 
- **Inverse of A Matrix** : 행렬 A가 정방행렬이고 A와 동일한 사이즈의 행렬 B일 때, $AB=BA=I$라면 $A$는 **invertible**(or **nonsingular**)이라 한다. 이 때 B는 A의 **inverse**이다. 만약 위 조건을 만족시키는 B가 존재하지 않다면 A는 **singular**하다고 한다.
  - invertible matrix는 unique inverse을 가진다.
  - $AA^{-1}=I$ and $A^{-1}A=I$
  - $(AB)^{-1}=B^{-1}A^{-1}$
- **Powers of A Matrix**
  - $(A^{-1})^{-1}= A$
  - $(A^n)^{-1}=A^{-n}=(A^{-1})^n$
  - nonzero scalar $k$일 때, $(kA)^{-1}=k^{-1}A^{-1}$
- **Matrix Polynomials** : 만약 A가 정방행렬 일 때,<br>
$$p(x)=a_0+a_1x+a_2x^2+...+a_mx^m$$ 이라면, $$p(A)=a_0I+a_1A+a_2A^2+...+a_mA^m$$ 이다.
- **Properties of The transpose**
  - $(A^T)^T= A$
  - $(A+B)^T=A^T+B^T$
  - $(A-B)^T=A^T-B^T$
  - $(kA)^T=kA^T$
  - $(AB)=B^TA^T$
  - 만약 행렬 $A$가 **invertible matrix**라면, 행렬 $A^T$ 또한 **invertible matrix**이다. 그리고 $(A^T)^{-1}=(A^{-1})^T$이다.
- **Properties of the Trace**
  - $tr(A^T)= tr(A)$
  - $tr(cA) = ctr(A)$
  - $tr(A+B)= tr(A)+tr(B)$
  - $tr(A-B)= tr(A)-tr(B)$
  - $tr(AB)=tr(BA)$
- **Transpose and Dot Product**
  - $A{\bf u \cdot v = u \cdot}A^T \bf v $
  - ${\bf u \cdot} A {\bf v = }A^T \bf u \cdot v$

<br>
<br>


# 3. Elementary Matrices

**elementary matrix** : 하나의 **elementary row operation**을 통해 Identy matrix$I_n$이 되는 행렬

**A Unifying Theorm**:만약 행렬 $A$가 $n\times n$ 정방행렬이라면, 아래 진술들은 모두 동일한 뜻이다.<br>

$A$의 RREF는 $I_n$이다. <br>$\Leftrightarrow$ $A$는 **elementary matrix**들로 표현 가능하다.<br> $\Leftrightarrow$ $A$의 역행렬이 존재한다. <br>$\Leftrightarrow$ $A\bf x = 0$는 오직 $trivial\ solution$이다.<br> $\Leftrightarrow$ 실수집합 $\mathbb{R}^n$에서 $A\bf x = b$는 모든 벡터 $\bf b$에 대해 일관된다.<br>$\Leftrightarrow$ 실수집합 $\mathbb{R}^n$에서 $A\bf x = b$는 모든 벡터 $\bf b$에 대해 정확한 한가지 Solution을 가진다.


본 장에서 역행렬을 푸는 여러 가지 방법을 소개했지만 이후 4장에서 **Cramer's Rule**을 통해 역행렬을 푸는 방법만 자세히 다루도록 하겠다.   

<br>
<br>


# 4. Subspaces and Linear Independence

1 장에서 $\mathbb{R}^2$에서의 직선과 $\mathbb{R}^3$에서의 평면을 각각
${\bf x = } t_1 {\bf v_1},\ \ {\bf x = } t_1 {\bf v_1}+t_2 {\bf v_2}$로 나타낼 수 있었다.<br>

이와 마찬가지로 $\mathbb{R}^n$에서의 geometric object(subspace)들은 아래와 같이 표현될 수 있다.
$${\bf x = } t_1 {\bf v_1} + t_2 {\bf v_2} + ... + t_s {\bf v_s}$$

또한 $\mathbb{R}^n$에서 S가 벡터들의 nonempty set$(S={\bf v_1, v_2, ..., v_s})$이라면, S는 **closed under scalar multiplication**, **closed under scalar addition**이 된다.<br>
이때, 만약 방정식 $c_1{\bf v_1} + c_2{\bf v_2} + ... + c_s{\bf v_s}= \bf 0$을 만족시키는 유일한 scalar들이 $c_1=0 , c_2=0 , ..., c_s=0$라면 이 때 S는 선형 독립이라 말한다. 만약 하나라도 0이 아니라면 선형 종속이라 말한다.

- 만약 $A$중 열벡터들이 선형 독립이라면 *homogeneous linear system* $A\bf x = 0$는 *trivial solution*을 가진다.
  - $$\begin{bmatrix}1 & 2&3 \\2&5&3\\1&0&8 \end{bmatrix} \begin{bmatrix}c_1 \\ c_2\\c_3 \end{bmatrix}=\begin{bmatrix}0 \\ 0\\0 \end{bmatrix}\\ solution : {\bf c=} \begin{bmatrix}0 \\ 0\\ 0 \end{bmatrix}$$ trival solution을 가진다.

<br>
<br>


# 5. Matrix Factorization; LU-Decomposition

**LU-decomposition** : 하삼각행렬 L와 상삼각행렬 U가 존재할 때, 정방 행렬 A = LU와 같이 정방행렬의 factorization을 *LU-decomposition* or *LU-factorization*이라 부른다.

- 만약 정방행렬 A가 행 교체 없이 Gausian Elimination에 의해 RREF(reduced to row echelon form)가 될 수 있다면,
A는 LU-decomposition을 가진다.
- 이 때 구해진 RREF는 상삼각행렬 $U$가 된다.
- 수행된 행 연산들이 $E_1, E_2,...E_k$일 때 $$L=E_1^{-1}E_2^{-1}...E_k^{-1}$$이 된다.

**Solving linear Systems by Factoriation**<br>
만약 정방 행렬 $A$, 상삼각행렬 $U$, 하삼각행렬 $L$가 존재할 때, <br>
$A=LU$로 표현할 수 있다.  

이 때 $A\bf x= b$의 해를 구하는 방법이 아래 나와 있다.

***Step 1.*** : $A\bf x = b$를 $LU \bf x = b$로 다시 작성한다. 

***Step 2.*** : $U \bf x = y$와 $L \bf y = b$가 되도록 하는 알려지지 않은 $\bf y$를 정의한다.

***Step 3.*** : $L \bf y=b$를 통해 $\bf y$의 해를 구한다.

***Step 4.*** : Step 3에서 구해진  $\bf y$와 $U \bf x = y$를 통해 $\bf x$를 구한다.

이 절차를 **LU-decomposition**의 method라 부른다.


**LDU-decomposition**<br>
L은 하삼각행렬, D가 대각행렬일 때, $$L=L'D$$로 표현될 수 있다.
예를 들어
$$\begin{bmatrix}a_{11} & 0&0 \\a_{21}&a_{22} &0 \\a_{31}&a_{32}&a_{33}\end{bmatrix} = 
\begin{bmatrix}1 & 0&0 \\a_{21}/a_{11}&1 &0 \\a_{31}/a_{11}&a_{32}/a_{22}&1\end{bmatrix}=
\begin{bmatrix}a_{11} & 0&0 \\0&a_{22} &0 \\0&0&a_{33}\end{bmatrix}
\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ L\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ L' \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ D$$
와 같이 표현될 수 있다.