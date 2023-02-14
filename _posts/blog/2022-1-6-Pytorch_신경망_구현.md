---
title: "Pytorch 신경망 구현"
category: DeepLearning
tags: [Pytorch, MLP, Deep Learning]
comments: true
date : 2022-01-06
categories: 
  - blog
excerpt: MLP
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

<br>



    

# PyTorch: Tensors (MLP 구현)

<br>

```python
import torch
print(torch.__version__)
```

    1.10.0+cu111

 - **'GPU'가 여러개 일때 cuda:'0'을 통해 GPU 선택 가능**

```python
dtype = torch.float
device = torch.device('cuda:0')
```



```python
N, D_in, H, D_out = 64, 1000, 100, 10
```


```python
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)
```

- N is batch size; D_in is input dimension
- H is hidden dimension; D_out is output dimension


```python
w1 = torch.randn(D_in, H, device = device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype =dtype)
```

 - **Randomly initialize weights**


```python
learning_rate = 1e-6
for i in range(500):
  # Forward pass : compute predicted y
  h = x.mm(w1) # (64, 1000) %*% (1000, 100) -> (64, 100)
  h_relu = h.clamp(min = 0) #relu 함수. 0이상이면 본인 값 0 이하면 0  
  y_pred = h_relu.mm(w2)

  #Compute and print loss
  loss = (y_pred - y).pow(2).sum().item() #제곱 후 더한다 scalr값으로 접근
  if i % 100 == 99:
    print(i, loss)
  
  #Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y) # loss의 편미분 값 제곱의 2 가 앞으로 나온다.
  grad_w2 = h_relu.t().mm(grad_y_pred) # transpose 후 행렬곱, 'mm'행렬곱
  grad_h_relu = grad_y_pred.mm(w2.t()) 
  grad_h = grad_h_relu.clone()
  grad_h[h < 0] = 0
  grad_w1 = x.t().mm(grad_h)

  # Update weights using gradient descent
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2 
```

    99 394.897705078125
    199 1.6938419342041016
    299 0.013515010476112366
    399 0.00030673673609271646
    499 4.544113471638411e-05
    
<br>

# PyTorch: Tensors and autograd

- 앞 부분은 동일하다. 
- Backprop 자동


```python
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H,D_out, device = device, dtype = dtype, requires_grad = True)
```


```python
learning_rate = 1e-6
for i in range(500):
  # Forward pass : compute predicted y
  y_pred = x.mm(w1).clamp(min=0).mm(w2)

  #Compute and print loss
  loss = (y_pred - y).pow(2).sum()#제곱 후 더한다 scalr값으로 접근
  if i % 100 == 99:
    print(i, loss.item())
  
  #Backprop to compute gradients of w1 and w2 with respect to loss
  loss.backward()

  # Update weights using gradient descent
  with torch.no_grad(): # 현재 gradient 고정
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad 

    w1.grad.zero_()   
    w2.grad.zero_()
```

    99 632.560791015625
    199 4.2360639572143555
    299 0.045063525438308716
    399 0.000844988040626049
    499 9.262723324354738e-05
    

<br>

# Pytorch: Defiining new autograd functions


```python
class MyReLU(torch.autograd.Function): #torch,autograd.function을 상속받음
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.clamp(min = 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0 ] = 0
    return grad_input
```

윗부분 동일


```python
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H,D_out, device = device, dtype = dtype, requires_grad = True)
```


```python
learning_rate = 1e-6
for i in range(500):
  #To apply our Function, we use Function.apply method. Me alias this as 'relu'.
  relu = MyReLU.apply

  # Forward pass : compute predicted y
  y_pred = relu(x.mm(w1)).mm(w2)

  #Compute and print loss
  loss = (y_pred - y).pow(2).sum() #제곱 후 더한다 scalr값으로 접근
  if i % 100 == 99:
    print(i, loss.item())
  
  #Backprop to compute gradients of w1 and w2 with respect to loss
  loss.backward()

  # Update weights using gradient descent
  with torch.no_grad(): # 현재 gradient 고정
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    w1.grad.zero_()   
    w2.grad.zero_()
```

    99 504.43798828125
    199 2.0029187202453613
    299 0.015105146914720535
    399 0.0003503029001876712
    499 5.5341926781693473e-05
    

<br>

# PyTorch:nn


```python
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H,D_out, device = device, dtype = dtype, requires_grad = True)
```


```python
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )
```


```python
criterion = torch.nn.MSELoss(reduction = 'sum')
learning_rate = 1e-4

for i in range(500):
  # Forward pass : compute predicted y
  model.to(device) # model과 tensor x 값의 같은 device에 존재해야한다.
  y_pred = model(x)

  #Compute and print loss
  loss = criterion(y_pred, y)
  if i % 100 == 99:
    print(i, loss.item())
  
  # Zero the gradients before running the backward pass. 모델의 weight의 grad를 초기화함
  model.zero_grad()

  #Backprop to compute gradients of w1 and w2 with respect to loss
  loss.backward()

  # Update weights using gradient descent
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad
```

    99 2.3959412574768066
    199 0.04334176331758499
    299 0.0026707835495471954
    399 0.0002706442610360682
    499 3.164863301208243e-05
    
<br>

# PyTorch: optim


```python
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H,D_out, device = device, dtype = dtype, requires_grad = True)
```


```python
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )
```


```python
criterion = torch.nn.MSELoss(reduction = 'sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(500):
  # Forward pass : compute predicted y
  model.to(device) # model과 tensor x 값의 같은 device에 존재해야한다.
  y_pred = model(x)

  #Compute and print loss
  loss = criterion(y_pred, y)
  if i % 100 == 99:
    print(i, loss.item())
  
  # Zero the gradients before running the backward pass. 모델의 weight의 grad를 초기화함
  optimizer.zero_grad()

  #Backprop to compute gradients of w1 and w2 with respect to loss
  loss.backward()

  optimizer.step()
```

    99 42.5167236328125
    199 0.262287437915802
    299 0.0005784527165815234
    399 2.036199930444127e-06
    499 4.341357051629302e-09
    
<br>

# PyTorch: Custom nn Modules


```python
class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()  #nn.Module 을 상속받기 위해 super 사용
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)


  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred
```


```python
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H,D_out, device = device, dtype = dtype, requires_grad = True)
```


```python
model = TwoLayerNet(D_in, H, D_out)
```


```python
criterion = torch.nn.MSELoss(reduction = 'sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(500):
  # Forward pass : compute predicted y
  model.to(device) # model과 tensor x 값의 같은 device에 존재해야한다.
  y_pred = model(x)

  #Compute and print loss
  loss = criterion(y_pred, y)
  if i % 100 == 99:
    print(i, loss.item())
  
  # Zero the gradients before running the backward pass. 모델의 weight의 grad를 초기화함
  optimizer.zero_grad()

  #Backprop to compute gradients of w1 and w2 with respect to loss
  loss.backward()

  optimizer.step()
```

    99 54.21607208251953
    199 0.536896288394928
    299 0.0016704383306205273
    399 1.1744625226128846e-05
    499 5.2012079976293535e-08
    
