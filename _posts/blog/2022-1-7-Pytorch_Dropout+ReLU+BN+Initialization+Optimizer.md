---
title: "Pytorch Dropout+ReLU+BN+Initialization+Optimizer"
category: DeepLearning
tags: [Pytorch, MLP, Deep Learning]
comments: true
date : 2022-01-07

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

# 1. Library & Data Load


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
```


    
![png](/images/2022-1-7-Pytorch_Dropout%2BReLU%2BBN%2BInitialization%2BOptimizer_files/2022-1-7-Pytorch_Dropout%2BReLU%2BBN%2BInitialization%2BOptimizer_7_0.png)
    

<br>

# 2. 모델링


```python
def weight_init(m):
  if isinstance(m, nn.Linear):              # MLP모델을 구성하고 있는 파라미터 중 nn.Linear에 해당하는 파라미터에만 적용
    init.kaiming_uniform_(m.weight.data)    #he_initialization
```


```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 10)
    self.dropout_prob = 0.5   # 50%의 노드에 대해 가중값을 계산하지 않는다.
    self.batch_norm1 = nn.BatchNorm1d(512)
    self.batch_norm2 = nn.BatchNorm1d(256)
  
  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = self.fc1(x)
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = F.dropout(x, training = self.training, p = self.dropout_prob) # "training = self.training"부분은 학습 상태일 때와 
                                                                      # 검증 상태에 따라 다르게 적용되기 위해 존재하는 파라미터 값이다.
    x = self.fc2(x)
    x = self.batch_norm2(x)
    x = F.relu(x)
    x = F.dropout(x, training = self.training, p = self.dropout_prob) 
    
    x = self.fc3(x)
    x = F.log_softmax(x, dim = 1)
    return x
```


```python
model = Net().to(DEVICE)
model.apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()
print(model)
```

    Net(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=10, bias=True)
      (batch_norm1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (batch_norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    


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


```python
for Epoch in range(1, EPOCHS +1):
  train(model, train_loader, optimizer, log_interval = 200)
  test_loss, test_accuracy = evaluate(model, test_loader)
  print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy : {:.2f} %\n".format(Epoch,
                                                                                  test_loss,
                                                                                  test_accuracy))  
```

    Train Epoch: 1 [0/60000(0%)] 	Train Loss : 3.135907
    Train Epoch: 1 [6400/60000(11%)] 	Train Loss : 0.528218
    Train Epoch: 1 [12800/60000(21%)] 	Train Loss : 0.816287
    Train Epoch: 1 [19200/60000(32%)] 	Train Loss : 0.673627
    Train Epoch: 1 [25600/60000(43%)] 	Train Loss : 0.315700
    Train Epoch: 1 [32000/60000(53%)] 	Train Loss : 0.208843
    Train Epoch: 1 [38400/60000(64%)] 	Train Loss : 0.097299
    Train Epoch: 1 [44800/60000(75%)] 	Train Loss : 0.077009
    Train Epoch: 1 [51200/60000(85%)] 	Train Loss : 0.152911
    Train Epoch: 1 [57600/60000(96%)] 	Train Loss : 0.425098
    
    [EPOCH: 1], 	Test Loss: 0.0044, 	Test Accuracy : 95.31 %
    
    Train Epoch: 2 [0/60000(0%)] 	Train Loss : 0.299352
    Train Epoch: 2 [6400/60000(11%)] 	Train Loss : 0.255701
    Train Epoch: 2 [12800/60000(21%)] 	Train Loss : 0.207824
    Train Epoch: 2 [19200/60000(32%)] 	Train Loss : 0.126038
    Train Epoch: 2 [25600/60000(43%)] 	Train Loss : 0.507558
    Train Epoch: 2 [32000/60000(53%)] 	Train Loss : 0.520747
    Train Epoch: 2 [38400/60000(64%)] 	Train Loss : 0.279233
    Train Epoch: 2 [44800/60000(75%)] 	Train Loss : 0.365085
    Train Epoch: 2 [51200/60000(85%)] 	Train Loss : 0.148094
    Train Epoch: 2 [57600/60000(96%)] 	Train Loss : 0.274170
    
    [EPOCH: 2], 	Test Loss: 0.0032, 	Test Accuracy : 96.87 %
    
    Train Epoch: 3 [0/60000(0%)] 	Train Loss : 0.405777
    Train Epoch: 3 [6400/60000(11%)] 	Train Loss : 0.427273
    Train Epoch: 3 [12800/60000(21%)] 	Train Loss : 0.061245
    Train Epoch: 3 [19200/60000(32%)] 	Train Loss : 0.301184
    Train Epoch: 3 [25600/60000(43%)] 	Train Loss : 0.249911
    Train Epoch: 3 [32000/60000(53%)] 	Train Loss : 0.120014
    Train Epoch: 3 [38400/60000(64%)] 	Train Loss : 0.160305
    Train Epoch: 3 [44800/60000(75%)] 	Train Loss : 0.163113
    Train Epoch: 3 [51200/60000(85%)] 	Train Loss : 0.625184
    Train Epoch: 3 [57600/60000(96%)] 	Train Loss : 0.128631
    
    [EPOCH: 3], 	Test Loss: 0.0031, 	Test Accuracy : 96.90 %
    
    Train Epoch: 4 [0/60000(0%)] 	Train Loss : 0.152833
    Train Epoch: 4 [6400/60000(11%)] 	Train Loss : 0.279276
    Train Epoch: 4 [12800/60000(21%)] 	Train Loss : 0.201038
    Train Epoch: 4 [19200/60000(32%)] 	Train Loss : 0.325278
    Train Epoch: 4 [25600/60000(43%)] 	Train Loss : 0.184978
    Train Epoch: 4 [32000/60000(53%)] 	Train Loss : 0.068118
    Train Epoch: 4 [38400/60000(64%)] 	Train Loss : 0.184181
    Train Epoch: 4 [44800/60000(75%)] 	Train Loss : 0.121350
    Train Epoch: 4 [51200/60000(85%)] 	Train Loss : 0.113255
    Train Epoch: 4 [57600/60000(96%)] 	Train Loss : 0.057423
    
    [EPOCH: 4], 	Test Loss: 0.0026, 	Test Accuracy : 97.35 %
    
    Train Epoch: 5 [0/60000(0%)] 	Train Loss : 0.058851
    Train Epoch: 5 [6400/60000(11%)] 	Train Loss : 0.135058
    Train Epoch: 5 [12800/60000(21%)] 	Train Loss : 0.030552
    Train Epoch: 5 [19200/60000(32%)] 	Train Loss : 0.183383
    Train Epoch: 5 [25600/60000(43%)] 	Train Loss : 0.058709
    Train Epoch: 5 [32000/60000(53%)] 	Train Loss : 0.352302
    Train Epoch: 5 [38400/60000(64%)] 	Train Loss : 0.149014
    Train Epoch: 5 [44800/60000(75%)] 	Train Loss : 0.193212
    Train Epoch: 5 [51200/60000(85%)] 	Train Loss : 0.115615
    Train Epoch: 5 [57600/60000(96%)] 	Train Loss : 0.136622
    
    [EPOCH: 5], 	Test Loss: 0.0023, 	Test Accuracy : 97.73 %
    
    Train Epoch: 6 [0/60000(0%)] 	Train Loss : 0.603114
    Train Epoch: 6 [6400/60000(11%)] 	Train Loss : 0.280477
    Train Epoch: 6 [12800/60000(21%)] 	Train Loss : 0.256615
    Train Epoch: 6 [19200/60000(32%)] 	Train Loss : 0.232010
    Train Epoch: 6 [25600/60000(43%)] 	Train Loss : 0.076881
    Train Epoch: 6 [32000/60000(53%)] 	Train Loss : 0.212273
    Train Epoch: 6 [38400/60000(64%)] 	Train Loss : 0.020975
    Train Epoch: 6 [44800/60000(75%)] 	Train Loss : 0.422267
    Train Epoch: 6 [51200/60000(85%)] 	Train Loss : 0.243079
    Train Epoch: 6 [57600/60000(96%)] 	Train Loss : 0.134959
    
    [EPOCH: 6], 	Test Loss: 0.0026, 	Test Accuracy : 97.67 %
    
    Train Epoch: 7 [0/60000(0%)] 	Train Loss : 0.253874
    Train Epoch: 7 [6400/60000(11%)] 	Train Loss : 0.056168
    Train Epoch: 7 [12800/60000(21%)] 	Train Loss : 0.115627
    Train Epoch: 7 [19200/60000(32%)] 	Train Loss : 0.187573
    Train Epoch: 7 [25600/60000(43%)] 	Train Loss : 0.074394
    Train Epoch: 7 [32000/60000(53%)] 	Train Loss : 0.065592
    Train Epoch: 7 [38400/60000(64%)] 	Train Loss : 0.182797
    Train Epoch: 7 [44800/60000(75%)] 	Train Loss : 0.427562
    Train Epoch: 7 [51200/60000(85%)] 	Train Loss : 0.155991
    Train Epoch: 7 [57600/60000(96%)] 	Train Loss : 0.047873
    
    [EPOCH: 7], 	Test Loss: 0.0023, 	Test Accuracy : 97.80 %
    
    Train Epoch: 8 [0/60000(0%)] 	Train Loss : 0.022977
    Train Epoch: 8 [6400/60000(11%)] 	Train Loss : 0.115374
    Train Epoch: 8 [12800/60000(21%)] 	Train Loss : 0.113944
    Train Epoch: 8 [19200/60000(32%)] 	Train Loss : 0.053478
    Train Epoch: 8 [25600/60000(43%)] 	Train Loss : 0.513141
    Train Epoch: 8 [32000/60000(53%)] 	Train Loss : 0.191641
    Train Epoch: 8 [38400/60000(64%)] 	Train Loss : 0.009619
    Train Epoch: 8 [44800/60000(75%)] 	Train Loss : 0.098782
    Train Epoch: 8 [51200/60000(85%)] 	Train Loss : 0.253856
    Train Epoch: 8 [57600/60000(96%)] 	Train Loss : 0.136868
    
    [EPOCH: 8], 	Test Loss: 0.0021, 	Test Accuracy : 98.03 %
    
    Train Epoch: 9 [0/60000(0%)] 	Train Loss : 0.030495
    Train Epoch: 9 [6400/60000(11%)] 	Train Loss : 0.763462
    Train Epoch: 9 [12800/60000(21%)] 	Train Loss : 0.147997
    Train Epoch: 9 [19200/60000(32%)] 	Train Loss : 0.091437
    Train Epoch: 9 [25600/60000(43%)] 	Train Loss : 0.027256
    Train Epoch: 9 [32000/60000(53%)] 	Train Loss : 0.038549
    Train Epoch: 9 [38400/60000(64%)] 	Train Loss : 0.220319
    Train Epoch: 9 [44800/60000(75%)] 	Train Loss : 0.099781
    Train Epoch: 9 [51200/60000(85%)] 	Train Loss : 0.162800
    Train Epoch: 9 [57600/60000(96%)] 	Train Loss : 0.118972
    
    [EPOCH: 9], 	Test Loss: 0.0022, 	Test Accuracy : 97.90 %
    
    Train Epoch: 10 [0/60000(0%)] 	Train Loss : 0.172032
    Train Epoch: 10 [6400/60000(11%)] 	Train Loss : 0.392355
    Train Epoch: 10 [12800/60000(21%)] 	Train Loss : 0.119749
    Train Epoch: 10 [19200/60000(32%)] 	Train Loss : 0.065733
    Train Epoch: 10 [25600/60000(43%)] 	Train Loss : 0.228694
    Train Epoch: 10 [32000/60000(53%)] 	Train Loss : 0.066152
    Train Epoch: 10 [38400/60000(64%)] 	Train Loss : 0.159296
    Train Epoch: 10 [44800/60000(75%)] 	Train Loss : 0.068501
    Train Epoch: 10 [51200/60000(85%)] 	Train Loss : 0.164852
    Train Epoch: 10 [57600/60000(96%)] 	Train Loss : 0.216015
    
    [EPOCH: 10], 	Test Loss: 0.0020, 	Test Accuracy : 98.10 %
    
    
