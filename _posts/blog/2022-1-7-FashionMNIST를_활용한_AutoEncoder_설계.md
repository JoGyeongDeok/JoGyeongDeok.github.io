---
title: "FashionMNIST를 활용한 AutoEncoder 설계"
category: DeepLearning
tags: [Pytorch, AutoEncoder, Deep Learning]
comments: true
date : 2022-01-07
categories: 
  - blog
excerpt: AutoEncoder
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

<br>

# 1.Library & Data Load


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
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print('Using PyTorch version : ', torch.__version__, 'Device : ', device)
```

    Using PyTorch version :  1.10.0+cu111 Device :  cuda
    


```python
BATCH_SIZE = 32
EPOCHS = 10
```


```python
train_dataset = datasets.FashionMNIST(root = "../data/FashionMNIST", 
                               train = True, 
                               download = True,
                               transform =transforms.ToTensor()
                               )
test_dataset = datasets.FashionMNIST(root = "../data/FashionMNIST", 
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

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/26421880 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/29515 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/4422102 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/5148 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/FashionMNIST/raw
    
    


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


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_7_0.png)
    
<br>

# 2. 모델링

AutoEncoder 모델 설계하기


```python
class AE(nn.Module):
  def __init__(self):
    super(AE, self).__init__()

    self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256,32),
    )

    self.decoder = nn.Sequential(
        nn.Linear(32, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 28 * 28),
    )
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded
```


```python
model = AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()
print(model)
```

    AE(
      (encoder): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=32, bias=True)
      )
      (decoder): Sequential(
        (0): Linear(in_features=32, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=784, bias=True)
      )
    )
    


```python
def train(model, train_loader, optimizer, log_interval):
  model.train()
  for batch_idx, (image, _) in enumerate(train_loader):
    image = image.view(-1, 28 * 28).to(device)
    target = image.view(-1, 28 * 28).to(device)
    optimizer.zero_grad()
    encoded, decoded = model(image)
    loss = criterion(decoded, target)
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
  real_image = []
  gen_image = []

  with torch.no_grad():
    for image, _ in test_loader:
      image = image.view(-1, 28 * 28).to(device)
      target = image.view(-1, 28 * 28).to(device)
      encoded, decoded = model(image)

      test_loss += criterion(decoded, image).item()
      real_image.append(image.to('cpu'))
      gen_image.append(decoded.to('cpu'))   
  test_loss /= len(test_loader.dataset)
  return test_loss, real_image, gen_image
```


```python
for Epoch in range(1, EPOCHS +1):
  train(model, train_loader, optimizer, log_interval = 200)
  test_loss, real_image, gen_image = evaluate(model, test_loader)
  print("\n[EPOCH: {}], \tTest Loss: {:.4f}".format(Epoch, test_loss))
  f, a = plt.subplots(2, 10, figsize = (10, 4))
  for i in range(10):
    img = np.reshape(real_image[0][i], (28, 28))
    a[0][i].imshow(img, cmap = 'gray_r')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

  for i in range(10):
    img = np.reshape(gen_image[0][i], (28, 28))
    a[1][i].imshow(img, cmap = 'gray_r')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
  plt.show()
```

    Train Epoch: 1 [0/60000(0%)] 	Train Loss : 0.213323
    Train Epoch: 1 [6400/60000(11%)] 	Train Loss : 0.025102
    Train Epoch: 1 [12800/60000(21%)] 	Train Loss : 0.022610
    Train Epoch: 1 [19200/60000(32%)] 	Train Loss : 0.019740
    Train Epoch: 1 [25600/60000(43%)] 	Train Loss : 0.019225
    Train Epoch: 1 [32000/60000(53%)] 	Train Loss : 0.017236
    Train Epoch: 1 [38400/60000(64%)] 	Train Loss : 0.013961
    Train Epoch: 1 [44800/60000(75%)] 	Train Loss : 0.015429
    Train Epoch: 1 [51200/60000(85%)] 	Train Loss : 0.019043
    Train Epoch: 1 [57600/60000(96%)] 	Train Loss : 0.016376
    
    [EPOCH: 1], 	Test Loss: 0.0005
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_1.png)
    


    Train Epoch: 2 [0/60000(0%)] 	Train Loss : 0.016465
    Train Epoch: 2 [6400/60000(11%)] 	Train Loss : 0.018139
    Train Epoch: 2 [12800/60000(21%)] 	Train Loss : 0.014978
    Train Epoch: 2 [19200/60000(32%)] 	Train Loss : 0.013692
    Train Epoch: 2 [25600/60000(43%)] 	Train Loss : 0.015178
    Train Epoch: 2 [32000/60000(53%)] 	Train Loss : 0.014941
    Train Epoch: 2 [38400/60000(64%)] 	Train Loss : 0.011518
    Train Epoch: 2 [44800/60000(75%)] 	Train Loss : 0.013371
    Train Epoch: 2 [51200/60000(85%)] 	Train Loss : 0.013450
    Train Epoch: 2 [57600/60000(96%)] 	Train Loss : 0.014339
    
    [EPOCH: 2], 	Test Loss: 0.0004
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_3.png)
    


    Train Epoch: 3 [0/60000(0%)] 	Train Loss : 0.011471
    Train Epoch: 3 [6400/60000(11%)] 	Train Loss : 0.012603
    Train Epoch: 3 [12800/60000(21%)] 	Train Loss : 0.010495
    Train Epoch: 3 [19200/60000(32%)] 	Train Loss : 0.011307
    Train Epoch: 3 [25600/60000(43%)] 	Train Loss : 0.010740
    Train Epoch: 3 [32000/60000(53%)] 	Train Loss : 0.014623
    Train Epoch: 3 [38400/60000(64%)] 	Train Loss : 0.011912
    Train Epoch: 3 [44800/60000(75%)] 	Train Loss : 0.010994
    Train Epoch: 3 [51200/60000(85%)] 	Train Loss : 0.009435
    Train Epoch: 3 [57600/60000(96%)] 	Train Loss : 0.010220
    
    [EPOCH: 3], 	Test Loss: 0.0004
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_5.png)
    


    Train Epoch: 4 [0/60000(0%)] 	Train Loss : 0.011250
    Train Epoch: 4 [6400/60000(11%)] 	Train Loss : 0.011909
    Train Epoch: 4 [12800/60000(21%)] 	Train Loss : 0.012980
    Train Epoch: 4 [19200/60000(32%)] 	Train Loss : 0.011037
    Train Epoch: 4 [25600/60000(43%)] 	Train Loss : 0.012296
    Train Epoch: 4 [32000/60000(53%)] 	Train Loss : 0.009452
    Train Epoch: 4 [38400/60000(64%)] 	Train Loss : 0.012458
    Train Epoch: 4 [44800/60000(75%)] 	Train Loss : 0.009762
    Train Epoch: 4 [51200/60000(85%)] 	Train Loss : 0.009361
    Train Epoch: 4 [57600/60000(96%)] 	Train Loss : 0.009820
    
    [EPOCH: 4], 	Test Loss: 0.0004
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_7.png)
    


    Train Epoch: 5 [0/60000(0%)] 	Train Loss : 0.009012
    Train Epoch: 5 [6400/60000(11%)] 	Train Loss : 0.009483
    Train Epoch: 5 [12800/60000(21%)] 	Train Loss : 0.009544
    Train Epoch: 5 [19200/60000(32%)] 	Train Loss : 0.011891
    Train Epoch: 5 [25600/60000(43%)] 	Train Loss : 0.012857
    Train Epoch: 5 [32000/60000(53%)] 	Train Loss : 0.009259
    Train Epoch: 5 [38400/60000(64%)] 	Train Loss : 0.009992
    Train Epoch: 5 [44800/60000(75%)] 	Train Loss : 0.010262
    Train Epoch: 5 [51200/60000(85%)] 	Train Loss : 0.011632
    Train Epoch: 5 [57600/60000(96%)] 	Train Loss : 0.013781
    
    [EPOCH: 5], 	Test Loss: 0.0003
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_9.png)
    


    Train Epoch: 6 [0/60000(0%)] 	Train Loss : 0.008445
    Train Epoch: 6 [6400/60000(11%)] 	Train Loss : 0.007757
    Train Epoch: 6 [12800/60000(21%)] 	Train Loss : 0.010105
    Train Epoch: 6 [19200/60000(32%)] 	Train Loss : 0.013357
    Train Epoch: 6 [25600/60000(43%)] 	Train Loss : 0.010602
    Train Epoch: 6 [32000/60000(53%)] 	Train Loss : 0.009949
    Train Epoch: 6 [38400/60000(64%)] 	Train Loss : 0.011849
    Train Epoch: 6 [44800/60000(75%)] 	Train Loss : 0.012378
    Train Epoch: 6 [51200/60000(85%)] 	Train Loss : 0.012695
    Train Epoch: 6 [57600/60000(96%)] 	Train Loss : 0.008380
    
    [EPOCH: 6], 	Test Loss: 0.0003
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_11.png)
    


    Train Epoch: 7 [0/60000(0%)] 	Train Loss : 0.011838
    Train Epoch: 7 [6400/60000(11%)] 	Train Loss : 0.009845
    Train Epoch: 7 [12800/60000(21%)] 	Train Loss : 0.010885
    Train Epoch: 7 [19200/60000(32%)] 	Train Loss : 0.009947
    Train Epoch: 7 [25600/60000(43%)] 	Train Loss : 0.010337
    Train Epoch: 7 [32000/60000(53%)] 	Train Loss : 0.010848
    Train Epoch: 7 [38400/60000(64%)] 	Train Loss : 0.010215
    Train Epoch: 7 [44800/60000(75%)] 	Train Loss : 0.010841
    Train Epoch: 7 [51200/60000(85%)] 	Train Loss : 0.009442
    Train Epoch: 7 [57600/60000(96%)] 	Train Loss : 0.009673
    
    [EPOCH: 7], 	Test Loss: 0.0003
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_13.png)
    


    Train Epoch: 8 [0/60000(0%)] 	Train Loss : 0.009594
    Train Epoch: 8 [6400/60000(11%)] 	Train Loss : 0.010143
    Train Epoch: 8 [12800/60000(21%)] 	Train Loss : 0.008687
    Train Epoch: 8 [19200/60000(32%)] 	Train Loss : 0.011797
    Train Epoch: 8 [25600/60000(43%)] 	Train Loss : 0.009236
    Train Epoch: 8 [32000/60000(53%)] 	Train Loss : 0.010680
    Train Epoch: 8 [38400/60000(64%)] 	Train Loss : 0.011123
    Train Epoch: 8 [44800/60000(75%)] 	Train Loss : 0.013803
    Train Epoch: 8 [51200/60000(85%)] 	Train Loss : 0.009494
    Train Epoch: 8 [57600/60000(96%)] 	Train Loss : 0.011025
    
    [EPOCH: 8], 	Test Loss: 0.0003
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_15.png)
    


    Train Epoch: 9 [0/60000(0%)] 	Train Loss : 0.009260
    Train Epoch: 9 [6400/60000(11%)] 	Train Loss : 0.009607
    Train Epoch: 9 [12800/60000(21%)] 	Train Loss : 0.008702
    Train Epoch: 9 [19200/60000(32%)] 	Train Loss : 0.009493
    Train Epoch: 9 [25600/60000(43%)] 	Train Loss : 0.009049
    Train Epoch: 9 [32000/60000(53%)] 	Train Loss : 0.012843
    Train Epoch: 9 [38400/60000(64%)] 	Train Loss : 0.009998
    Train Epoch: 9 [44800/60000(75%)] 	Train Loss : 0.011692
    Train Epoch: 9 [51200/60000(85%)] 	Train Loss : 0.011123
    Train Epoch: 9 [57600/60000(96%)] 	Train Loss : 0.009878
    
    [EPOCH: 9], 	Test Loss: 0.0003
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_17.png)
    


    Train Epoch: 10 [0/60000(0%)] 	Train Loss : 0.009773
    Train Epoch: 10 [6400/60000(11%)] 	Train Loss : 0.011040
    Train Epoch: 10 [12800/60000(21%)] 	Train Loss : 0.009801
    Train Epoch: 10 [19200/60000(32%)] 	Train Loss : 0.009314
    Train Epoch: 10 [25600/60000(43%)] 	Train Loss : 0.009573
    Train Epoch: 10 [32000/60000(53%)] 	Train Loss : 0.010427
    Train Epoch: 10 [38400/60000(64%)] 	Train Loss : 0.007872
    Train Epoch: 10 [44800/60000(75%)] 	Train Loss : 0.011646
    Train Epoch: 10 [51200/60000(85%)] 	Train Loss : 0.008943
    Train Epoch: 10 [57600/60000(96%)] 	Train Loss : 0.009305
    
    [EPOCH: 10], 	Test Loss: 0.0003
    


    
![png](/images/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_files/2022-1-7-FashionMNIST%EB%A5%BC_%ED%99%9C%EC%9A%A9%ED%95%9C_AutoEncoder_%EC%84%A4%EA%B3%84_14_19.png)
    

