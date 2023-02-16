---
title: "Fully Convolutional Networks for Semantic Segmentation (2)"
tags: [Pytorch, Deep Learning, Computer Vision, Semantic Segmentation]
comments: true
excerpt: Semantic Segmentation
date : 2022-03-05
categories: 
  - PaperReview
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Computer Vision
---
<style>
      .language-plainte{
    background-color : rgba($color: #ffffff, $alpha: 1.0);
    margin-bottom: 3em;
  }
  .scrol{
    height : 700px;
    overflow:auto;
  }

</style>


<br>

# IV. Implement

## 1. Data load & Library load


```python
!wget -O '/content/data' http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
!tar -xvf data
```


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import cv2
```


```python
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
```


```python
n_class    = 21
batch_size = 10
epochs     = 100
h = 448
w = 448
```


```python
if torch.cuda.is_available():
    device = 'cuda'
else : 
    device ='cpu'
```
<br> 

## 2. Model

VGG-16 fine-tuning


```python
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```


```python
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```


```python
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}
```


```python
model = 'vgg19'
```

**vgg_19 model**
- 19 weight layers


```python
 vgg_layers = make_layers(cfg[model])
 print(vgg_layers)
```

    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace=True)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace=True)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace=True)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace=True)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace=True)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    


```python
vgg_ranges = ranges[model]
print(vgg_ranges)
```

    ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
    

구간의 끝에서 MaxPolling을 한다.


```python
class VGGNet(VGG):
    def __init__(self, pretrained=True, requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(vgg_layers)
        self.ranges = vgg_ranges

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        '''
        get the output of each maxpooling layer (5 maxpool in VGG net)
        layer를 5개로 분할 (maxpooling 하는 곳마다 분할) 
            => pooling시 image 줄어드니까 이걸 다시 업샘플링 하려는듯
        '''
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x     

        return output
```


```python
class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        
        # encoder
        self.pretrained_net = pretrained_net
        
        #decoder
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        # skip Architecture
        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
```

- Skip Architecture 
 - 구간별 Maxpooling + Upsampling 결과를 합한다.


```python
train_path = pd.read_csv("/content/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt",header=None)
test_path = pd.read_csv("/content/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/val.txt",header=None)


# image_path
train_img_path = []
for i in train_path[0].values:
    train_img_path.append("/content/TrainVal/VOCdevkit/VOC2011/JPEGImages/" + i + '.jpg')

test_img_path = []
for i in test_path[0].values:
    test_img_path.append("/content/TrainVal/VOCdevkit/VOC2011/JPEGImages/" + i + '.jpg')
```


```python
# anno_path
train_anno_path = []
for i in train_path[0].values:
    train_anno_path.append("/content/TrainVal/VOCdevkit/VOC2011/SegmentationClass/" + i + '.png')

test_anno_path = []
for i in test_path[0].values:
    test_anno_path.append("/content/TrainVal/VOCdevkit/VOC2011/SegmentationClass/" + i + '.png')
```

VOC2011의 Segmentation 데이터의 png 이미지는 존재하지만 
클래스별 색상 정보는 존재하지 않아 임의로 구한다.

- 0.1 \* R_color + 0.521 \* G_color + 0.192 \* B_color


```python
# img_value = []
# for i in tqdm(temp_train_anno_path):
#     img = cv2.imread(i)
#     for i in range(len(img)):
#         for j in range(len(img[0])):
#             temp = 0.1*img[i,j,0] + 0.521*img[i,j,1] + 0.192*img[i,j,2]
#             if temp > 0 :
#                 img_value.append(temp)
# np.unique(img_value)
'''
# 178.912가 테두리
array([ 12.288,  12.8  ,  24.576,  25.088,  33.344,  36.864,  37.376,
        46.144,  49.664,  57.92 ,  66.688,  78.976,  79.488,  91.264,
        91.776, 100.032, 103.552, 104.064, 116.352, 124.608, 178.912])
'''

```




    '\n# 178.912가 테두리\narray([ 12.288,  12.8  ,  24.576,  25.088,  33.344,  36.864,  37.376,\n        46.144,  49.664,  57.92 ,  66.688,  78.976,  79.488,  91.264,\n        91.776, 100.032, 103.552, 104.064, 116.352, 124.608, 178.912])\n'




```python
num_range = [ 0, 12.288,  12.8  ,  24.576,  25.088,  33.344,  36.864,  37.376,
        46.144,  49.664,  57.92 ,  66.688,  78.976,  79.488,  91.264,
        91.776, 100.032, 103.552, 104.064, 116.352, 124.608]
```


```python
# image show
def draw(img):
    plt.figure(figsize = (10,30))
    for i, n in enumerate([0, 5, 7]):
        oi = torch.argmax(img[n].to('cpu').squeeze(), dim=0)
        om = oi.detach().numpy()
        rgb = decode_segmap(om)
        plt.subplot(1,3,i+1)
        plt.imshow(rgb)
    plt.show()
```


```python
# 이미지 그리는 함수
def decode_segmap(image, nc=n_class):
  label_colors = np.array([(0, 0, 0), # 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb
```


```python
# mAP 계산 함수
def mAP_fn(y_hat, y):
    mAP = []
    for c in range(n_class):
        pred_inds = y_hat == c
        target_inds = y == c
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
    
        for i in range(50,100, 5):
            if float(intersection) / max(union, 1) > i * 0.01 :
                mAP.append(1) # If there is no ground truth, do not include in evaluation
            else:
                mAP.append(0)
    return mAP

# pixel 정확도 계산 함수
def pixel_accuracy(y_hat, y):
    correct = (y_hat == y).sum()
    total = (y == y).sum()
    return correct/total
```


```python
class Custom_dataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.h = h
        self.w = w
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self,idx):

        # 이미지 변환
        img = self.img_path[idx]
        img = cv2.imread(img)/255.
        img = cv2.resize(img,(self.h, self.w))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(img)


        # 라벨 변환
        # label_img(batch_size, h, w, 3) => label_img(batch_size, n_class, h, w) 형태로 변환
        label = self.label_path[idx]
        label_img = cv2.imread(label)
        label_img = cv2.resize(label_img,(self.h, self.w))
        new_img = torch.zeros(label_img.shape[0] * label_img.shape[1] * 21).reshape(21,label_img.shape[0], label_img.shape[1])
        temp = 0.1*label_img[:,:,0] + 0.521*label_img[:,:,1] + 0.192*label_img[:,:,2]
        for i in range(len(num_range)):
            if i < len(num_range):
                new_img[i,:,:] = (transforms.ToTensor()(temp)[0] == num_range[i]) 
        label_img = new_img

        return img, label_img
```

**Model Load**


```python
vgg_model = VGGNet(requires_grad=True).to(device)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class).to(device)

from torch.optim import lr_scheduler
# Define training parameters
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=0.0001, momentum=0, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

```

    Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to /root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
    


      0%|          | 0.00/548M [00:00<?, ?B/s]



```python
# 파라미터 개수
pytorch_total_params = sum(p.numel() for p in fcn_model.parameters() if p.requires_grad)
print(pytorch_total_params)
```

    23954069
    

**DataLoader**


```python
# Train
train_dataset = Custom_dataset(train_img_path, train_anno_path)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Validation 
valid_dataset = Custom_dataset(test_img_path, test_anno_path)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
```


```python
def model_train():
    # scheduler.step()
    train_loss = 0
    fcn_model.train()
    for iter,batch in enumerate(train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.float32, device=device)
        pred = fcn_model(x)      
        loss = criterion(pred, y)
        loss.backward()

        optimizer.step()      
        train_loss += loss.item()/len(train_loader)
    return train_loss


def model_eval(epoch) :
    fcn_model.eval()
    valid_loss = 0
    total_mAP = []
    pixel_accs = []
    with torch.no_grad():
        for iter, batch in enumerate(valid_loader):
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.float32, device=device)
            output = fcn_model(x)
            loss = criterion(output, y)
            valid_loss += loss.item()/len(valid_loader)
            
            # image show
            if epoch % 5 == 0:
                if iter == 0 :
                    draw(output)
                
            #mAP, pixel_accuracy 
            output = output.data.cpu().numpy()
            N, _, h, w = output.shape 
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
            target = torch.tensor(batch[1].argmax(1), dtype=torch.float32, device=device)
            for y_hat, y in zip(pred, target.to('cpu').numpy()):
                total_mAP.append(mAP_fn(y_hat, y))
                pixel_accs.append(pixel_accuracy(y_hat, y))
    # Pixel accuracy
    pixel_acc = np.array(pixel_accs).mean()
    
    # Mean mAP
    total_mAPs = np.array(total_mAP).T
    mAP = np.nanmean(np.nanmean(total_mAP, axis=1))

    return valid_loss, pixel_acc, mAP
```


```python
for batch in valid_loader:
    x = torch.tensor(batch[0], dtype=torch.float32, device=device)
    y = torch.tensor(batch[1], dtype=torch.float32, device=device)
    pred = nn.functional.sigmoid(fcn_model(x))
    break
```


```python
plt.figure(figsize = (20,60))
plt.subplot(1,3,1)
plt.imshow(cv2.imread(test_img_path[0]))
plt.subplot(1,3,2)
plt.imshow(cv2.imread(test_img_path[5]))
plt.subplot(1,3,3)
plt.imshow(cv2.imread(test_img_path[7]))
```




    <matplotlib.image.AxesImage at 0x7f67ad354dd0>




    
![png](/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_35_1.png)
    



```python
draw(y)
```


    
![png](/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_36_0.png)
    

<br>

## 3. Training


```python
best = 1
for epoch in range(epochs):
    scheduler.step()

    start=time.time()
    train_loss = model_train()

    eval_loss, paccuracy, mAP = model_eval(epoch)

    if best > eval_loss:
        best = eval_loss
    print("{} epoch score,   eval loss : {},   time elapsed : {}\n".format(epoch, eval_loss, time.time() - start))
    print("best score : {}\n".format(best))
    print( "Pixel accuracy : {:.5f}".format(paccuracy),
              "Mean mAP : {:.5f}".format(mAP))
    print("==============================\n")
```

<div class = 'scrol'>
<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_0.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>0 epoch score,   eval loss : 0.7178982315318925,   time elapsed : 241.66198635101318

best score : 0.7178982315318925

Pixel accuracy : 0.30074 Mean mAP : 0.00528
==============================

1 epoch score,   eval loss : 0.6619830865945135,   time elapsed : 240.5474236011505

best score : 0.6619830865945135

Pixel accuracy : 0.50298 Mean mAP : 0.01512
==============================

2 epoch score,   eval loss : 0.6436454510050161,   time elapsed : 239.4425733089447

best score : 0.6436454510050161

Pixel accuracy : 0.47292 Mean mAP : 0.01354
==============================

3 epoch score,   eval loss : 0.6180395196591109,   time elapsed : 238.40908098220825

best score : 0.6180395196591109

Pixel accuracy : 0.55352 Mean mAP : 0.01732
==============================

4 epoch score,   eval loss : 0.5651034648929324,   time elapsed : 237.66926980018616

best score : 0.5651034648929324

Pixel accuracy : 0.73754 Mean mAP : 0.03038
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_2.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>5 epoch score,   eval loss : 0.536102440740381,   time elapsed : 237.24283647537231

best score : 0.536102440740381

Pixel accuracy : 0.71512 Mean mAP : 0.03138
==============================

6 epoch score,   eval loss : 0.5178359661783493,   time elapsed : 236.54765009880066

best score : 0.5178359661783493

Pixel accuracy : 0.62283 Mean mAP : 0.02511
==============================

7 epoch score,   eval loss : 0.8486483400421482,   time elapsed : 236.49466514587402

best score : 0.5178359661783493

Pixel accuracy : 0.16001 Mean mAP : 0.00070
==============================

8 epoch score,   eval loss : 0.44291923088686797,   time elapsed : 235.9113450050354

best score : 0.44291923088686797

Pixel accuracy : 0.74320 Mean mAP : 0.03266
==============================

9 epoch score,   eval loss : 0.42381186330957066,   time elapsed : 236.72971606254578

best score : 0.42381186330957066

Pixel accuracy : 0.63047 Mean mAP : 0.02557
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_4.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>10 epoch score,   eval loss : 0.38310617341526926,   time elapsed : 236.9712359905243

best score : 0.38310617341526926

Pixel accuracy : 0.70987 Mean mAP : 0.02959
==============================

11 epoch score,   eval loss : 0.3471113491271225,   time elapsed : 236.41412138938904

best score : 0.3471113491271225

Pixel accuracy : 0.73506 Mean mAP : 0.03437
==============================

12 epoch score,   eval loss : 0.3159190770238639,   time elapsed : 235.39537358283997

best score : 0.3159190770238639

Pixel accuracy : 0.80542 Mean mAP : 0.03622
==============================

13 epoch score,   eval loss : 0.29627867468765795,   time elapsed : 235.42185235023499

best score : 0.29627867468765795

Pixel accuracy : 0.65226 Mean mAP : 0.02755
==============================

14 epoch score,   eval loss : 0.2869952957012823,   time elapsed : 236.63488125801086

best score : 0.2869952957012823

Pixel accuracy : 0.56076 Mean mAP : 0.01801
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_6.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>15 epoch score,   eval loss : 0.24047080041574578,   time elapsed : 236.50986194610596

best score : 0.24047080041574578

Pixel accuracy : 0.73603 Mean mAP : 0.03295
==============================

16 epoch score,   eval loss : 0.22134679875203536,   time elapsed : 235.29431223869324

best score : 0.22134679875203536

Pixel accuracy : 0.72685 Mean mAP : 0.03348
==============================

17 epoch score,   eval loss : 0.2155027627678854,   time elapsed : 234.138578414917

best score : 0.2155027627678854

Pixel accuracy : 0.61898 Mean mAP : 0.02091
==============================

18 epoch score,   eval loss : 0.17580746965748925,   time elapsed : 235.54726696014404

best score : 0.17580746965748925

Pixel accuracy : 0.76055 Mean mAP : 0.03546
==============================

19 epoch score,   eval loss : 0.15687567501195848,   time elapsed : 235.1361322402954

best score : 0.15687567501195848

Pixel accuracy : 0.73649 Mean mAP : 0.03431
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_8.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>20 epoch score,   eval loss : 0.13700008359072458,   time elapsed : 235.14089441299438

best score : 0.13700008359072458

Pixel accuracy : 0.78662 Mean mAP : 0.03856
==============================

21 epoch score,   eval loss : 0.12497481537450637,   time elapsed : 233.5440411567688

best score : 0.12497481537450637

Pixel accuracy : 0.76943 Mean mAP : 0.03838
==============================

22 epoch score,   eval loss : 0.1422548733784684,   time elapsed : 234.46710395812988

best score : 0.12497481537450637

Pixel accuracy : 0.80738 Mean mAP : 0.03208
==============================

23 epoch score,   eval loss : 0.10834715109584582,   time elapsed : 234.6178023815155

best score : 0.10834715109584582

Pixel accuracy : 0.78982 Mean mAP : 0.03400
==============================

24 epoch score,   eval loss : 0.11611943891538042,   time elapsed : 234.59283423423767

best score : 0.10834715109584582

Pixel accuracy : 0.67101 Mean mAP : 0.02729
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_10.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>25 epoch score,   eval loss : 0.090947254061965,   time elapsed : 237.0232207775116

best score : 0.090947254061965

Pixel accuracy : 0.82170 Mean mAP : 0.03904
==============================

26 epoch score,   eval loss : 0.08642006638859,   time elapsed : 236.98462295532227

best score : 0.08642006638859

Pixel accuracy : 0.82043 Mean mAP : 0.03631
==============================

27 epoch score,   eval loss : 0.092587448723082,   time elapsed : 236.6958875656128

best score : 0.08642006638859

Pixel accuracy : 0.82626 Mean mAP : 0.03619
==============================

28 epoch score,   eval loss : 0.2560650797427766,   time elapsed : 235.55009150505066

best score : 0.08642006638859

Pixel accuracy : 0.24022 Mean mAP : 0.00162
==============================

29 epoch score,   eval loss : 0.08757972876940458,   time elapsed : 236.69172191619873

best score : 0.08642006638859

Pixel accuracy : 0.69619 Mean mAP : 0.03040
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_12.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>30 epoch score,   eval loss : 0.066161598045645,   time elapsed : 237.42317175865173

best score : 0.066161598045645

Pixel accuracy : 0.78961 Mean mAP : 0.03882
==============================

31 epoch score,   eval loss : 0.0920455172391874,   time elapsed : 237.43141889572144

best score : 0.066161598045645

Pixel accuracy : 0.68829 Mean mAP : 0.02954
==============================

32 epoch score,   eval loss : 0.06304706127515859,   time elapsed : 237.1986062526703

best score : 0.06304706127515859

Pixel accuracy : 0.79514 Mean mAP : 0.03951
==============================

33 epoch score,   eval loss : 0.07512970584710796,   time elapsed : 237.58603167533875

best score : 0.06304706127515859

Pixel accuracy : 0.72118 Mean mAP : 0.03164
==============================

34 epoch score,   eval loss : 0.06017713792555568,   time elapsed : 237.43538165092468

best score : 0.06017713792555568

Pixel accuracy : 0.78311 Mean mAP : 0.03784
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_14.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>35 epoch score,   eval loss : 0.059438517350437384,   time elapsed : 237.52762079238892

best score : 0.059438517350437384

Pixel accuracy : 0.77877 Mean mAP : 0.03697
==============================

36 epoch score,   eval loss : 0.08852791573320118,   time elapsed : 237.17932152748108

best score : 0.059438517350437384

Pixel accuracy : 0.74797 Mean mAP : 0.03027
==============================

37 epoch score,   eval loss : 0.059932751930318766,   time elapsed : 236.7956509590149

best score : 0.059438517350437384

Pixel accuracy : 0.82333 Mean mAP : 0.03965
==============================

38 epoch score,   eval loss : 0.057104336879482236,   time elapsed : 236.91790390014648

best score : 0.057104336879482236

Pixel accuracy : 0.79046 Mean mAP : 0.03831
==============================

39 epoch score,   eval loss : 0.05626014746459467,   time elapsed : 237.54847741127014

best score : 0.05626014746459467

Pixel accuracy : 0.78402 Mean mAP : 0.03789
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_16.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>40 epoch score,   eval loss : 0.05263682797418109,   time elapsed : 237.8654248714447

best score : 0.05263682797418109

Pixel accuracy : 0.80603 Mean mAP : 0.04078
==============================

41 epoch score,   eval loss : 0.05207437541269296,   time elapsed : 237.74000668525696

best score : 0.05207437541269296

Pixel accuracy : 0.81444 Mean mAP : 0.04175
==============================

42 epoch score,   eval loss : 0.05176435070045825,   time elapsed : 236.80525493621826

best score : 0.05176435070045825

Pixel accuracy : 0.80833 Mean mAP : 0.04141
==============================

43 epoch score,   eval loss : 0.06690562732650766,   time elapsed : 237.0335454940796

best score : 0.05176435070045825

Pixel accuracy : 0.80727 Mean mAP : 0.03964
==============================

44 epoch score,   eval loss : 0.05689868475643121,   time elapsed : 237.42520689964294

best score : 0.05176435070045825

Pixel accuracy : 0.78381 Mean mAP : 0.03889
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_18.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>45 epoch score,   eval loss : 0.05627184497591643,   time elapsed : 237.58590364456177

best score : 0.05176435070045825

Pixel accuracy : 0.79326 Mean mAP : 0.04058
==============================

46 epoch score,   eval loss : 0.06605513430466609,   time elapsed : 236.72612166404724

best score : 0.05176435070045825

Pixel accuracy : 0.72504 Mean mAP : 0.03474
==============================

47 epoch score,   eval loss : 0.05040174213770243,   time elapsed : 236.87144303321838

best score : 0.05040174213770243

Pixel accuracy : 0.81848 Mean mAP : 0.04252
==============================

48 epoch score,   eval loss : 0.05617471327007352,   time elapsed : 237.2989420890808

best score : 0.05040174213770243

Pixel accuracy : 0.83600 Mean mAP : 0.04171
==============================

49 epoch score,   eval loss : 0.053576983478186395,   time elapsed : 237.2237627506256

best score : 0.05040174213770243

Pixel accuracy : 0.82332 Mean mAP : 0.04163
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_20.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>50 epoch score,   eval loss : 0.06716973332056243,   time elapsed : 237.3336877822876

best score : 0.05040174213770243

Pixel accuracy : 0.83735 Mean mAP : 0.04032
==============================

51 epoch score,   eval loss : 0.052568263008392256,   time elapsed : 236.78479647636414

best score : 0.05040174213770243

Pixel accuracy : 0.81006 Mean mAP : 0.04143
==============================

52 epoch score,   eval loss : 0.05184695660136639,   time elapsed : 236.86836576461792

best score : 0.05040174213770243

Pixel accuracy : 0.81326 Mean mAP : 0.04207
==============================

53 epoch score,   eval loss : 0.11699245890070288,   time elapsed : 236.74232721328735

best score : 0.05040174213770243

Pixel accuracy : 0.59510 Mean mAP : 0.02334
==============================

54 epoch score,   eval loss : 0.05507521043598118,   time elapsed : 237.1274299621582

best score : 0.05040174213770243

Pixel accuracy : 0.80951 Mean mAP : 0.04207
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_22.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>55 epoch score,   eval loss : 0.05524524713733365,   time elapsed : 236.9862220287323

best score : 0.05040174213770243

Pixel accuracy : 0.83906 Mean mAP : 0.04383
==============================

56 epoch score,   eval loss : 0.06026830158329436,   time elapsed : 237.25112462043762

best score : 0.05040174213770243

Pixel accuracy : 0.80863 Mean mAP : 0.04039
==============================

57 epoch score,   eval loss : 0.05404031329921313,   time elapsed : 237.3176829814911

best score : 0.05040174213770243

Pixel accuracy : 0.80580 Mean mAP : 0.04108
==============================

58 epoch score,   eval loss : 0.05592775777248401,   time elapsed : 237.0596535205841

best score : 0.05040174213770243

Pixel accuracy : 0.83400 Mean mAP : 0.04339
==============================

59 epoch score,   eval loss : 0.05457758981667993,   time elapsed : 237.00527691841125

best score : 0.05040174213770243

Pixel accuracy : 0.83410 Mean mAP : 0.04288
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_24.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>60 epoch score,   eval loss : 0.05413613910786807,   time elapsed : 236.9829077720642

best score : 0.05040174213770243

Pixel accuracy : 0.80638 Mean mAP : 0.04136
==============================

61 epoch score,   eval loss : 0.060561432735994465,   time elapsed : 236.88742852210999

best score : 0.05040174213770243

Pixel accuracy : 0.75022 Mean mAP : 0.03789
==============================

62 epoch score,   eval loss : 0.06334032988109223,   time elapsed : 236.4282808303833

best score : 0.05040174213770243

Pixel accuracy : 0.83103 Mean mAP : 0.04144
==============================

63 epoch score,   eval loss : 0.05944986435185586,   time elapsed : 236.974933385849

best score : 0.05040174213770243

Pixel accuracy : 0.78637 Mean mAP : 0.03998
==============================

64 epoch score,   eval loss : 0.06143085199541278,   time elapsed : 236.53927159309387

best score : 0.05040174213770243

Pixel accuracy : 0.81822 Mean mAP : 0.04301
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_26.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>65 epoch score,   eval loss : 0.05019922158680855,   time elapsed : 236.974267244339

best score : 0.05019922158680855

Pixel accuracy : 0.82812 Mean mAP : 0.04415
==============================

66 epoch score,   eval loss : 0.06073784191227916,   time elapsed : 237.1444399356842

best score : 0.05019922158680855

Pixel accuracy : 0.83814 Mean mAP : 0.04204
==============================

67 epoch score,   eval loss : 0.05965266840731993,   time elapsed : 237.22307896614075

best score : 0.05019922158680855

Pixel accuracy : 0.81144 Mean mAP : 0.04086
==============================

68 epoch score,   eval loss : 0.05388594558462502,   time elapsed : 236.73139214515686

best score : 0.05019922158680855

Pixel accuracy : 0.83474 Mean mAP : 0.04438
==============================

69 epoch score,   eval loss : 0.08115607895888391,   time elapsed : 237.58847975730896

best score : 0.05019922158680855

Pixel accuracy : 0.73731 Mean mAP : 0.03497
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_28.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>70 epoch score,   eval loss : 0.05781047571716567,   time elapsed : 237.8951563835144

best score : 0.05019922158680855

Pixel accuracy : 0.78503 Mean mAP : 0.04083
==============================

71 epoch score,   eval loss : 0.056797722786931085,   time elapsed : 237.23341369628906

best score : 0.05019922158680855

Pixel accuracy : 0.82074 Mean mAP : 0.04286
==============================

72 epoch score,   eval loss : 0.055418342144028944,   time elapsed : 237.99950432777405

best score : 0.05019922158680855

Pixel accuracy : 0.84128 Mean mAP : 0.04500
==============================

73 epoch score,   eval loss : 0.5618845040776898,   time elapsed : 237.01102876663208

best score : 0.05019922158680855

Pixel accuracy : 0.03605 Mean mAP : 0.00017
==============================

74 epoch score,   eval loss : 0.06444542815110516,   time elapsed : 237.69644165039062

best score : 0.05019922158680855

Pixel accuracy : 0.79356 Mean mAP : 0.04141
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_30.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>75 epoch score,   eval loss : 0.061300998902879655,   time elapsed : 237.72181034088135

best score : 0.05019922158680855

Pixel accuracy : 0.80727 Mean mAP : 0.04003
==============================

76 epoch score,   eval loss : 0.05329450086823533,   time elapsed : 237.24029564857483

best score : 0.05019922158680855

Pixel accuracy : 0.82964 Mean mAP : 0.04366
==============================

77 epoch score,   eval loss : 0.06170900649989823,   time elapsed : 237.39747214317322

best score : 0.05019922158680855

Pixel accuracy : 0.80903 Mean mAP : 0.04119
==============================

78 epoch score,   eval loss : 0.0564126601010295,   time elapsed : 237.27731895446777

best score : 0.05019922158680855

Pixel accuracy : 0.84171 Mean mAP : 0.04500
==============================

79 epoch score,   eval loss : 0.05411140412823963,   time elapsed : 236.86781764030457

best score : 0.05019922158680855

Pixel accuracy : 0.82108 Mean mAP : 0.04370
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_32.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>80 epoch score,   eval loss : 0.07080968665624301,   time elapsed : 237.34644722938538

best score : 0.05019922158680855

Pixel accuracy : 0.72713 Mean mAP : 0.03606
==============================

81 epoch score,   eval loss : 0.05516451003495606,   time elapsed : 237.56378316879272

best score : 0.05019922158680855

Pixel accuracy : 0.83558 Mean mAP : 0.04524
==============================

82 epoch score,   eval loss : 0.05482077189455077,   time elapsed : 236.7994179725647

best score : 0.05019922158680855

Pixel accuracy : 0.83377 Mean mAP : 0.04494
==============================

83 epoch score,   eval loss : 0.05405946047643996,   time elapsed : 236.29342555999756

best score : 0.05019922158680855

Pixel accuracy : 0.83390 Mean mAP : 0.04515
==============================

84 epoch score,   eval loss : 0.057823608180374984,   time elapsed : 236.97596549987793

best score : 0.05019922158680855

Pixel accuracy : 0.84011 Mean mAP : 0.04439
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_34.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>85 epoch score,   eval loss : 0.06111668964980967,   time elapsed : 237.5672435760498

best score : 0.05019922158680855

Pixel accuracy : 0.80194 Mean mAP : 0.04156
==============================

86 epoch score,   eval loss : 0.0566813734725916,   time elapsed : 236.2354302406311

best score : 0.05019922158680855

Pixel accuracy : 0.82161 Mean mAP : 0.04376
==============================

87 epoch score,   eval loss : 0.06325172674509558,   time elapsed : 236.15085983276367

best score : 0.05019922158680855

Pixel accuracy : 0.78035 Mean mAP : 0.03930
==============================

88 epoch score,   eval loss : 0.056199040704606386,   time elapsed : 236.91179609298706

best score : 0.05019922158680855

Pixel accuracy : 0.84521 Mean mAP : 0.04544
==============================

89 epoch score,   eval loss : 0.060648024032291546,   time elapsed : 236.8065779209137

best score : 0.05019922158680855

Pixel accuracy : 0.82861 Mean mAP : 0.04378
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_36.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>90 epoch score,   eval loss : 0.05827999407691614,   time elapsed : 236.68064618110657

best score : 0.05019922158680855

Pixel accuracy : 0.80702 Mean mAP : 0.04224
==============================

91 epoch score,   eval loss : 0.05737729549374695,   time elapsed : 236.5483682155609

best score : 0.05019922158680855

Pixel accuracy : 0.84375 Mean mAP : 0.04560
==============================

92 epoch score,   eval loss : 0.056048106784666235,   time elapsed : 236.41613626480103

best score : 0.05019922158680855

Pixel accuracy : 0.84530 Mean mAP : 0.04600
==============================

93 epoch score,   eval loss : 0.057354241227065866,   time elapsed : 236.9008662700653

best score : 0.05019922158680855

Pixel accuracy : 0.83146 Mean mAP : 0.04394
==============================

94 epoch score,   eval loss : 0.05561174947901494,   time elapsed : 236.65051198005676

best score : 0.05019922158680855

Pixel accuracy : 0.84092 Mean mAP : 0.04521
==============================
</code></pre></div></div>

<p><img src="/images/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_files/2022-3-18-pytorch_Fully_Convolutional_Networks_for_Semantic_Segmentation_38_38.png" alt="png" /></p>

<div class="language-plainte highlighter-rouge"><div class="highlight"><pre class="highlight"><code>95 epoch score,   eval loss : 0.05528645654807667,   time elapsed : 237.61715745925903

best score : 0.05019922158680855

Pixel accuracy : 0.83989 Mean mAP : 0.04549
==============================

96 epoch score,   eval loss : 0.05694875355610354,   time elapsed : 236.72872924804688

best score : 0.05019922158680855

Pixel accuracy : 0.82715 Mean mAP : 0.04352
==============================

97 epoch score,   eval loss : 0.10319366236217326,   time elapsed : 235.93873381614685

best score : 0.05019922158680855

Pixel accuracy : 0.59625 Mean mAP : 0.02451
==============================

98 epoch score,   eval loss : 0.05805794088934946,   time elapsed : 236.15517926216125

best score : 0.05019922158680855

Pixel accuracy : 0.84998 Mean mAP : 0.04560
==============================

99 epoch score,   eval loss : 0.05806795251555743,   time elapsed : 236.07272505760193

best score : 0.05019922158680855

Pixel accuracy : 0.83604 Mean mAP : 0.04475
==============================
</code></pre></div></div>
</div>
    
    

<br>

# Reference
[1] J Long, E Shelhamer, T Darrell. (2015). Fully convolutional networks for semantic segmentation

[2] https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204