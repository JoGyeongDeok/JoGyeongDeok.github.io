---
title: "Pytorch GAN"
tags: [Pytorch, Computer Vision, Deep Learning]
comments: true
date : 2022-02-16
categories: 
  - blog
excerpt: GAN
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---

<br>

# 1. Data & Library Load


```python
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
```


```python
transforms_train = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./dataset/MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/train-images-idx3-ubyte.gz to ./dataset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./dataset/MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/train-labels-idx1-ubyte.gz to ./dataset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./dataset/MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/t10k-images-idx3-ubyte.gz to ./dataset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./dataset/MNIST/raw
    
    

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:566: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    

# 2. 생성자(Generator) 및 판별자(Discriminator) 모델 정의


```python
latent_dim = 100


# 생성자(Generator) 클래스 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 하나의 블록(block) 정의
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                # 배치 정규화(batch normalization) 수행(차원 동일)
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 생성자 모델은 연속적인 여러 개의 블록을 가짐
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img
```


```python
# 판별자(Discriminator) 클래스 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    # 이미지에 대한 판별 결과를 반환
    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)

        return output
```

<br>

# 3. 모델 학습 및 샘플링


```python
# 생성자(generator)와 판별자(discriminator) 초기화
generator = Generator()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()

# 손실 함수(loss function)
adversarial_loss = nn.BCELoss() #이진분류
adversarial_loss.cuda()

# 학습률(learning rate) 설정
lr = 0.0002

# 생성자와 판별자를 위한 최적화 함수
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
```


```python
g_loss_list = []
d_loss_list = []

import time

n_epochs = 300 # 학습의 횟수(epoch) 설정
sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # 진짜(real): 1
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # 가짜(fake): 0

        real_imgs = imgs.cuda()

        """ 생성자(generator)를 학습합니다. """
        optimizer_G.zero_grad()

        # 랜덤 노이즈(noise) 샘플링
        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()

        # 이미지 생성
        generated_imgs = generator(z)

        # 생성자(generator)의 손실(loss) 값 계산
        g_loss = adversarial_loss(discriminator(generated_imgs), real)

        # 생성자(generator) 업데이트
        g_loss.backward()
        optimizer_G.step()

        """ 판별자(discriminator)를 학습합니다. """
        optimizer_D.zero_grad()

        # 판별자(discriminator)의 손실(loss) 값 계산
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # 판별자(discriminator) 업데이트
        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            # 생성된 이미지 중에서 25개만 선택하여 5 X 5 격자 이미지에 출력
            save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

    # 하나의 epoch이 끝날 때마다 로그(log) 출력
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time}]")
    d_loss_list.append(d_loss.item())
    g_loss_list.append(g_loss.item())
```

    [Epoch 0/300] [D loss: 0.447488] [G loss: 0.939881] [Elapsed time: 19.825135946273804]
    [Epoch 1/300] [D loss: 0.368966] [G loss: 1.950788] [Elapsed time: 32.68967604637146]
    [Epoch 2/300] [D loss: 0.390395] [G loss: 0.813334] [Elapsed time: 45.64992713928223]
    [Epoch 3/300] [D loss: 0.348491] [G loss: 1.157016] [Elapsed time: 58.72316002845764]
    [Epoch 4/300] [D loss: 0.211269] [G loss: 1.552729] [Elapsed time: 71.51849317550659]
    [Epoch 5/300] [D loss: 0.282206] [G loss: 2.202995] [Elapsed time: 84.53256750106812]
    [Epoch 6/300] [D loss: 0.183141] [G loss: 2.069922] [Elapsed time: 98.31056046485901]
    [Epoch 7/300] [D loss: 0.308191] [G loss: 2.611833] [Elapsed time: 113.28064513206482]
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-2f338270b786> in <module>
         42 
         43         # 판별자(discriminator) 업데이트
    ---> 44         d_loss.backward()
         45         optimizer_D.step()
         46 
    

    /usr/local/lib/python3.7/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
        394                 create_graph=create_graph,
        395                 inputs=inputs)
    --> 396         torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
        397 
        398     def register_hook(self, hook):
    

    /usr/local/lib/python3.7/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        173     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        174         tensors, grad_tensors_, retain_graph, create_graph, inputs,
    --> 175         allow_unreachable=True, accumulate_grad=True)  # Calls into the C++ engine to run the backward pass
        176 
        177 def grad(
    

    KeyboardInterrupt: 



```python
from matplotlib import pylab as plt

plt.figure()
plt.plot(d_loss_list, label = 'd_loss')
plt.plot(g_loss_list, label = 'g_loss')
plt.show()
```


```python
from IPython.display import Image
Image('10000.png')
```


```python
Image('20000.png')
```




    
![png](/images/2022-2-16-Pytorch_GAN_files/2022-2-16-Pytorch_GAN_12_0.png)
    




```python
Image('30000.png')
```




    
![png](/images/2022-2-16-Pytorch_GAN_files/2022-2-16-Pytorch_GAN_13_0.png)
    




```python
Image('40000.png')
```




    
![png](/images/2022-2-16-Pytorch_GAN_files/2022-2-16-Pytorch_GAN_14_0.png)
    




```python
Image('98000.png')
```




    
![png](/images/2022-2-16-Pytorch_GAN_files/2022-2-16-Pytorch_GAN_15_0.png)
    


