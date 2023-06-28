---
title: "Linear Algebra - 1(Vectors)"
category: Mathematics
tags: [Linear Algebra, Mathematics]
comments: true
date : 2023-05-03
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

# 1. Dot Product and Orthogonality

${\bf u}\ =\ (u_1,\ u_2,\ ...,\ u_n)$, ${\bf v}\ =\ (v_1,\ v_2,\ ...\ v_n)$ and ${\bf w}\ =\ (w_1,\ w_2,\ ...\ w_n)$들이 실수 공간 $\mathbb{R}^n$에 속할 때,
<br>

**Norm of A Vector**<br>
$\parallel\ {\bf v} \parallel\ =\ \sqrt{v_1^2+v_2^2+v_3^2+...+v_n^2}$
- $\parallel\ {\bf v} \parallel\ \geq\ 0$
- 만약 $\bf v\ =\ 0$이라면, $\parallel\ {\bf v} \parallel\ =\ 0$
- $\parallel\ k {\bf v} \parallel\ =$ $\mid k\mid$$\parallel\ \bf v \parallel$
<br>

**Unit Vector**<br>
- 길이가 1인 벡터
- $\bf u\ =\ \frac{1}{\parallel\ v \parallel}v$
<br>

**The Standard Unit Vector**<br>
- 좌표축 양의 방향에 있는 단위벡터
- ${\bf i}\ =\ (1,0)$ and ${\bf j}\ =\ (0, 1)$
- ${\bf e_1}\ =\ (1,0,0,...,0),\ {\bf e_2}\ =\ (0,1,0,...,0),\ ...,\ {\bf e_n}\ =\ (0,0,0,...,1)$일 때,
    - ${\bf v}\ =\ (v_1,\ v_2,...,\ v_n)\ =\ v_1{\bf e_1}\ +\ v_2{\bf e_2}\ +\ ...\ +\ v_n{\bf e_n}$
<br>

**Dot Product**<br>
${\bf u \cdot v }\ =\ u_1v_1\ +\ u_2v_2\ +\ ...\ +\ u_n v_n$ 
- ${\bf 0 \cdot v}\ =\ 0$
- $\bf u \cdot v\ =\ v \cdot u$
- $\bf u \cdot (v+w)\ =\ u \cdot v + u \cdot w$
- $\bf u \cdot (v-w)\ =\ u \cdot v - u \cdot w$
- $k ({\bf u \cdot v})\ =\ (k \bf u) \cdot v$
- ${\bf v \cdot v} \ =\ \parallel\bf v\parallel$
<br>

**Angle Between Vectors in $\mathbb{R}^2$ and $\mathbb{R}^3$**<br>
<img src = '/images/Linear Algebra/1_1.jpeg' height = 300 width = 350>

$\bf u,\ v$가 nonzero vectors 일때, $cos \theta\ =\ \frac{\bf u \cdot v}{\parallel u\parallel \parallel v\parallel }$<br>
**Proof**<br>
${\bf \parallel v - u \parallel^2\ =\ (v-u)\cdot (v-u)}$<br> $ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ {\bf =}\ \bf{v \cdot v- u \cdot v - v \cdot u + u \cdot u}$<br>  $ {\bf \Leftrightarrow}\ \parallel \bf{v} \parallel^2 - 2 {\bf u \cdot v + \parallel u }\parallel ^2 {\bf =}\ \parallel \bf{v} \parallel^2 - 2 {\bf \parallel u\parallel \parallel v\parallel cos\theta + \parallel u }\parallel ^2$<br> $\therefore\ cos \theta\ =\ \frac{\bf u \cdot v}{\parallel u\parallel \parallel v\parallel }$
 - $-1 \leq \frac{\bf u \cdot v}{\parallel u\parallel \parallel v\parallel } \leq 1$
 - $(\bf u \cdot v)^2 \leq \parallel u\parallel \parallel v\parallel$ : Cauchy-Schwarz Inequality
 - $\bf \parallel u+v \parallel \leq \parallel u \parallel + \parallel v \parallel$ : Triangle Inequality for Vectors
<br>

**Orthogonality**<br>
${\bf u \cdot v}\ = 0$ 일 때 $\bf u,\ v$는 서로 Orthogonality한다고 말한다.
- $\bf u, v$가 서로 직교할 때$(cos\theta\ =\ 0)$,
    - $\bf \parallel u + v \parallel^2\  =\ \parallel u \parallel^2 + \parallel v \parallel^2$

<br>
<br>

# 2. Vector Equations of Lines and Planes

**Vector and Parametric Equations of Lines**<br>
실수 집합 $\mathbb{R}^2$직선의 일반 방정식의 형태는 $Ax + By = C$(A and B not both zero)였다. <br>

실수 집합 $\mathbb{R}^2,\ \mathbb{R}^3$에서 직선의 방정식$\bf x$을 nonzero vector $\bf v$로 표한할 수 있다.<br>
${\bf x\ =\ x_0\ +}t {\bf v}\ (-\infty<t<+\infty)$<br>
여기서 ${\bf x\ =\ x_0\ +}t {\bf v}$는 $\bf x_0$에 의해 ${\bf x\ =\ }t {\bf v}$가 **translation**된것이다.

<img src = '/images/Linear Algebra/1_2.png' height = 300 width = 800>
<br>

**Point-Normal Equations of Planes**<br>
평면 $W$가 uniquely determined되기 위해서는 평면 $W$위에 specifying point $\bf x_0$와 two nonzero vectors $\bf v_1, v_2$가 필요하다. 이 때 평면 $W$위의 점 $\bf x$는 아래와 같다.<br>
${\bf x\ =\ x_0+} t_1 {\bf v_1} + t_2 {\bf v_2}\ \ (-\infty<t_1, t_2 < +\infty)$

<img src = '/images/Linear Algebra/1_3.png' height = 300 width = 600>
<br>

**Lines And Planes In $\mathbb{R}^2$**<br>
만약 $\bf x_0$가 실수집합 $\mathbb{R}^n$의 벡터이고, $\bf v$가 실수집합 $\mathbb{R}^n$의 nonzero vector일 때,
- line through $\bf x_0$ that is parallel to $\bf v$
    - ${\bf x\ =\ x_0\ +}t {\bf v}\ (-\infty<t<+\infty)$
- the plane through $\bf x_0$ that is parallel to $\bf v_1, v_2$ to be the set of all vectors $\bf x$ in $\mathbb{R}^n$
    - ${\bf x\ =\ x_0+} t_1 {\bf v_1} + t_2 {\bf v_2}\ \ (-\infty<t_1, t_2 < +\infty)$

