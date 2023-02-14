---
title: "MNIST를 이용한 MLP 설계"
tags: [Pytorch, MLP, Deep Learning,]
comments: true
date : 2022-01-02
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


## 1.Library & Data Load


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
```


```python
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')
print('Using PyTorch version : ', torch.__version__, '\nDevice : ', DEVICE)
```

    Using PyTorch version :  1.10.0+cu111 
    Device :  cuda
    


```python
BATCH_SIZE = 32
EPOCHS = 10
```


```python
train_dataset = datasets.MNIST(root = "../data/MNIST", 
                               train = True, 
                               download = True,
                               transform =transforms.ToTensor()
                               )
test_dataset = datasets.MNIST(root = "../data/MNIST", 
                               train = False, 
                               download = True,
                               transform =transforms.ToTensor()
                               )

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True
                                           )
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = False
                                           )
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw
    
    


```python
for (X_train, y_train) in train_loader:
  print('X_train : ', X_train.size(), 'type : ', X_train.type())
  print('y_train : ', y_train.size(), 'type : ', X_train.type())
  break
```

    X_train :  torch.Size([32, 1, 28, 28]) type :  torch.FloatTensor
    y_train :  torch.Size([32]) type :  torch.FloatTensor
    


```python
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
  plt.subplot(1, 10, i+1)
  plt.axis('off')
  plt.imshow(X_train[i, :, :, :].numpy().reshape(28,28), cmap = 'gray_r')
  plt.title('Class : ' + str(y_train[i].item()))
```


    
![png](/images/2022-1-2-MNIST%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_MLP_%EC%84%A4%EA%B3%84_files/2022-1-2-MNIST%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_MLP_%EC%84%A4%EA%B3%84_7_0.png)
    

<br><br>

## 2. 모델링

 - **MLP(Multi Layer Perceptron) 모델 설계**


```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(28*28, 512)
    self.fc2 = nn.Linear(512,256)
    self.fc3 = nn.Linear(256,10)
  
  def forward(self, x):
    x = x.view(-1, 28*28)
    x = self.fc1(x)
    x = F.sigmoid(x)
    x = self.fc2(x)
    x = F.sigmoid(x)
    x = self.fc3(x)
    x = F.log_softmax(x, dim = 1)
    return x
```

 - **Optimizer, Objective Function 설정**


```python
model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()
print(model)
```

    Net(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=10, bias=True)
    )
    

 - **MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의**


```python
def train(model, train_loader, optimizer, log_interval):
  model.train()
  for batch_idx, (image, label) in enumerate(train_loader):
    image = image.to(DEVICE)
    label = label.to(DEVICE)
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print("Train Epoch: {} [{}/{}({:.0f}%)] \tTrain Loss : {:.6f}".format(Epoch,
                                                                            batch_idx * len(image),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
```

 - **학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의**


```python
def evaluate(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for image, label in test_loader:
      image = image.to(DEVICE)
      label = label.to(DEVICE)
      output = model(image)
      test_loss += criterion(output, label).item()
      prediction = output.max(1, keepdim = True)[1]
      correct += prediction.eq(label.view_as(prediction)).sum().item()
    
  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, test_accuracy
```

 - **MLP 학습을 실행하면서 Train, Test set의 Loss 및 Test set Accuracy를 확인**


```python
for Epoch in range(1, EPOCHS +1):
  train(model, train_loader, optimizer, log_interval = 200)
  test_loss, test_accuracy = evaluate(model, test_loader)
  print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy : {:.2f} %\n".format(Epoch,
                                                                                  test_loss,
                                                                                  test_accuracy))
```

    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1806: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    

    Train Epoch: 1 [0/60000(0%)] 	Train Loss : 2.321163
    Train Epoch: 1 [6400/60000(11%)] 	Train Loss : 2.297261
    Train Epoch: 1 [12800/60000(21%)] 	Train Loss : 2.294299
    Train Epoch: 1 [19200/60000(32%)] 	Train Loss : 2.309930
    Train Epoch: 1 [25600/60000(43%)] 	Train Loss : 2.304135
    Train Epoch: 1 [32000/60000(53%)] 	Train Loss : 2.261675
    Train Epoch: 1 [38400/60000(64%)] 	Train Loss : 2.289791
    Train Epoch: 1 [44800/60000(75%)] 	Train Loss : 2.302188
    Train Epoch: 1 [51200/60000(85%)] 	Train Loss : 2.233261
    Train Epoch: 1 [57600/60000(96%)] 	Train Loss : 2.203007
    
    [EPOCH: 1], 	Test Loss: 0.0698, 	Test Accuracy : 31.45 %
    
    Train Epoch: 2 [0/60000(0%)] 	Train Loss : 2.207758
    Train Epoch: 2 [6400/60000(11%)] 	Train Loss : 2.184944
    Train Epoch: 2 [12800/60000(21%)] 	Train Loss : 2.172096
    Train Epoch: 2 [19200/60000(32%)] 	Train Loss : 2.102351
    Train Epoch: 2 [25600/60000(43%)] 	Train Loss : 1.933385
    Train Epoch: 2 [32000/60000(53%)] 	Train Loss : 1.953243
    Train Epoch: 2 [38400/60000(64%)] 	Train Loss : 1.770699
    Train Epoch: 2 [44800/60000(75%)] 	Train Loss : 1.584548
    Train Epoch: 2 [51200/60000(85%)] 	Train Loss : 1.518090
    Train Epoch: 2 [57600/60000(96%)] 	Train Loss : 1.307722
    
    [EPOCH: 2], 	Test Loss: 0.0388, 	Test Accuracy : 64.73 %
    
    Train Epoch: 3 [0/60000(0%)] 	Train Loss : 1.156514
    Train Epoch: 3 [6400/60000(11%)] 	Train Loss : 1.116772
    Train Epoch: 3 [12800/60000(21%)] 	Train Loss : 1.022202
    Train Epoch: 3 [19200/60000(32%)] 	Train Loss : 1.095805
    Train Epoch: 3 [25600/60000(43%)] 	Train Loss : 0.955300
    Train Epoch: 3 [32000/60000(53%)] 	Train Loss : 0.953761
    Train Epoch: 3 [38400/60000(64%)] 	Train Loss : 1.043167
    Train Epoch: 3 [44800/60000(75%)] 	Train Loss : 1.007626
    Train Epoch: 3 [51200/60000(85%)] 	Train Loss : 0.663740
    Train Epoch: 3 [57600/60000(96%)] 	Train Loss : 0.643852
    
    [EPOCH: 3], 	Test Loss: 0.0230, 	Test Accuracy : 78.14 %
    
    Train Epoch: 4 [0/60000(0%)] 	Train Loss : 0.820724
    Train Epoch: 4 [6400/60000(11%)] 	Train Loss : 0.378877
    Train Epoch: 4 [12800/60000(21%)] 	Train Loss : 0.889468
    Train Epoch: 4 [19200/60000(32%)] 	Train Loss : 0.445477
    Train Epoch: 4 [25600/60000(43%)] 	Train Loss : 0.908844
    Train Epoch: 4 [32000/60000(53%)] 	Train Loss : 0.615338
    Train Epoch: 4 [38400/60000(64%)] 	Train Loss : 0.311315
    Train Epoch: 4 [44800/60000(75%)] 	Train Loss : 0.741198
    Train Epoch: 4 [51200/60000(85%)] 	Train Loss : 0.353237
    Train Epoch: 4 [57600/60000(96%)] 	Train Loss : 0.773765
    
    [EPOCH: 4], 	Test Loss: 0.0171, 	Test Accuracy : 84.01 %
    
    Train Epoch: 5 [0/60000(0%)] 	Train Loss : 0.471269
    Train Epoch: 5 [6400/60000(11%)] 	Train Loss : 0.858984
    Train Epoch: 5 [12800/60000(21%)] 	Train Loss : 0.374166
    Train Epoch: 5 [19200/60000(32%)] 	Train Loss : 0.460735
    Train Epoch: 5 [25600/60000(43%)] 	Train Loss : 0.344983
    Train Epoch: 5 [32000/60000(53%)] 	Train Loss : 0.588109
    Train Epoch: 5 [38400/60000(64%)] 	Train Loss : 0.580059
    Train Epoch: 5 [44800/60000(75%)] 	Train Loss : 0.549942
    Train Epoch: 5 [51200/60000(85%)] 	Train Loss : 0.378287
    Train Epoch: 5 [57600/60000(96%)] 	Train Loss : 0.677744
    
    [EPOCH: 5], 	Test Loss: 0.0142, 	Test Accuracy : 86.70 %
    
    Train Epoch: 6 [0/60000(0%)] 	Train Loss : 0.456661
    Train Epoch: 6 [6400/60000(11%)] 	Train Loss : 0.474878
    Train Epoch: 6 [12800/60000(21%)] 	Train Loss : 0.352067
    Train Epoch: 6 [19200/60000(32%)] 	Train Loss : 0.466520
    Train Epoch: 6 [25600/60000(43%)] 	Train Loss : 0.466449
    Train Epoch: 6 [32000/60000(53%)] 	Train Loss : 0.551689
    Train Epoch: 6 [38400/60000(64%)] 	Train Loss : 0.418935
    Train Epoch: 6 [44800/60000(75%)] 	Train Loss : 0.574251
    Train Epoch: 6 [51200/60000(85%)] 	Train Loss : 0.268617
    Train Epoch: 6 [57600/60000(96%)] 	Train Loss : 0.468736
    
    [EPOCH: 6], 	Test Loss: 0.0128, 	Test Accuracy : 88.04 %
    
    Train Epoch: 7 [0/60000(0%)] 	Train Loss : 0.268330
    Train Epoch: 7 [6400/60000(11%)] 	Train Loss : 0.309484
    Train Epoch: 7 [12800/60000(21%)] 	Train Loss : 0.303733
    Train Epoch: 7 [19200/60000(32%)] 	Train Loss : 0.524566
    Train Epoch: 7 [25600/60000(43%)] 	Train Loss : 0.566025
    Train Epoch: 7 [32000/60000(53%)] 	Train Loss : 0.488985
    Train Epoch: 7 [38400/60000(64%)] 	Train Loss : 0.335191
    Train Epoch: 7 [44800/60000(75%)] 	Train Loss : 0.167177
    Train Epoch: 7 [51200/60000(85%)] 	Train Loss : 0.586831
    Train Epoch: 7 [57600/60000(96%)] 	Train Loss : 0.327236
    
    [EPOCH: 7], 	Test Loss: 0.0118, 	Test Accuracy : 88.97 %
    
    Train Epoch: 8 [0/60000(0%)] 	Train Loss : 0.964730
    Train Epoch: 8 [6400/60000(11%)] 	Train Loss : 0.211401
    Train Epoch: 8 [12800/60000(21%)] 	Train Loss : 0.516455
    Train Epoch: 8 [19200/60000(32%)] 	Train Loss : 0.333291
    Train Epoch: 8 [25600/60000(43%)] 	Train Loss : 0.338095
    Train Epoch: 8 [32000/60000(53%)] 	Train Loss : 0.232396
    Train Epoch: 8 [38400/60000(64%)] 	Train Loss : 0.463446
    Train Epoch: 8 [44800/60000(75%)] 	Train Loss : 0.698152
    Train Epoch: 8 [51200/60000(85%)] 	Train Loss : 0.259675
    Train Epoch: 8 [57600/60000(96%)] 	Train Loss : 0.488693
    
    [EPOCH: 8], 	Test Loss: 0.0112, 	Test Accuracy : 89.51 %
    
    Train Epoch: 9 [0/60000(0%)] 	Train Loss : 0.363097
    Train Epoch: 9 [6400/60000(11%)] 	Train Loss : 0.199268
    Train Epoch: 9 [12800/60000(21%)] 	Train Loss : 0.404107
    Train Epoch: 9 [19200/60000(32%)] 	Train Loss : 0.244276
    Train Epoch: 9 [25600/60000(43%)] 	Train Loss : 0.297371
    Train Epoch: 9 [32000/60000(53%)] 	Train Loss : 0.361722
    Train Epoch: 9 [38400/60000(64%)] 	Train Loss : 0.360184
    Train Epoch: 9 [44800/60000(75%)] 	Train Loss : 0.133858
    Train Epoch: 9 [51200/60000(85%)] 	Train Loss : 0.350550
    Train Epoch: 9 [57600/60000(96%)] 	Train Loss : 0.375616
    
    [EPOCH: 9], 	Test Loss: 0.0109, 	Test Accuracy : 89.80 %
    
    Train Epoch: 10 [0/60000(0%)] 	Train Loss : 0.565450
    Train Epoch: 10 [6400/60000(11%)] 	Train Loss : 0.257261
    Train Epoch: 10 [12800/60000(21%)] 	Train Loss : 0.144566
    Train Epoch: 10 [19200/60000(32%)] 	Train Loss : 0.175128
    Train Epoch: 10 [25600/60000(43%)] 	Train Loss : 0.264306
    Train Epoch: 10 [32000/60000(53%)] 	Train Loss : 0.400938
    Train Epoch: 10 [38400/60000(64%)] 	Train Loss : 0.141663
    Train Epoch: 10 [44800/60000(75%)] 	Train Loss : 0.374536
    Train Epoch: 10 [51200/60000(85%)] 	Train Loss : 0.303158
    Train Epoch: 10 [57600/60000(96%)] 	Train Loss : 0.376532
    
    [EPOCH: 10], 	Test Loss: 0.0105, 	Test Accuracy : 90.27 %
    
    
