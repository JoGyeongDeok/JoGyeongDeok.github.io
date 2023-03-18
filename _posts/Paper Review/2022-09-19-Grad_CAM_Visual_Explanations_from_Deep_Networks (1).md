---
title: "Grad-CAM Visual Explanations from Deep Networks (1)"
tags: [Pytorch, Computer Vision, Deep Learning, Explanable AI]
comments: true
excerpt: CAM
date : 2022-09-19
categories: 
  - PaperReview
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : Computer Vision
---

<style scoped>
    .long {
        height : 300px;
        overflow : auto;
    }
</style>


<br>

# I. CAM(Class Activation Map)

기존의 Learning deep features for discriminative localization(Zhou et al.)논문에서 Class Activation Mapping(CAM) 기법이 소개되었다. 
CNN기반의 Architecture에서 Convolution Layer를 거치고 바로 Fully Connected Layer를 사용하는 것이 아니라 GAP (Global Average Pooling)을 거친 후 Fully Connected Layer를 사용한다. 여기서 GAP은 Image의 값이 C x H x W일 때 C의 픽셀값을 평균 취한 것이다.

$$
GAP(x) = \frac{1}{WH}\sum^{W-1}_{w=0}\sum^{H-1}_{h=0}x[:,w,h]
$$


<img src = 'https://drive.google.com/uc?id=1W3x6OjnaAKErWlie8htz_QaJhmRk-4-1' height = 500 width = 700>

## CAM 계산

CAM을 구하기 위해서는 마지막 Convolutional Layer의 Output(= GAP의 input)과 바로 뒤의 Fully-Connected Layer의 Weight가 필요하다. 

$f_k(x,y)$: GAP의 input의 $k$번째 unit의 $(x, y)$좌표에 해당하는 값

$w_{k}^{c}$ : FC Layer에서 input의 $k$번째 unit과 output의 $c$번째 (class)에 대응되는 weight

$$
S_c = \sum_{k}w^c_{k}\sum_{x,y}f_k(x,y) = \sum_{k}\sum_{x,y}w^c_kf_k(x, y) = \sum_{x,y}\sum_{k}w^c_kf_k(x, y)
\\ 
M_c(x, y) = \sum_kw^c_kf_k(x,y) 라고 정의
$$

$M_c(x,y)$를 GAP의 input에서 좌표(x, y)가 class $c$에 얼마나 영향을 주는지와 관련된 수치라고 생각할 수 있다. 그리고 CAM의 결과는 $M_c(x, y)$로 이루어진 행렬을 원래 이미지 크기에 맞게 upscaling한 결과로 나타내어진다.

<br>

## CAM 구현


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from glob import glob
from tqdm import tqdm
import time
import cv2
from google.colab.patches import cv2_imshow

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import vgg19, resnet101

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms.functional import to_pil_image


import warnings
warnings.simplefilter(action='ignore') 

```


```python
if torch.cuda.is_available():
    device = 'cuda'
else : 
    device ='cpu'

batch_size = 32
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                      std = [ 0.229, 0.224, 0.225 ])
])

invTrans = transforms.Compose([ 
        transforms.Resize(128),
        transforms.Normalize(mean = [ 0., 0., 0. ],
        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
        std = [ 1., 1., 1. ]),
                               ])

# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
(h, w) = (128, 128)
(cX, cY) = (w // 2, h // 2)
 
# 이미지를 중심으로 -90도 회전
M_90 = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
```


```python
'''
CAM은 모델 하단부를 변경해야한다.
'''

class CAM_Model(nn.Module):
    def __init__(self, model, img_size, nc):
        super(CAM_Model, self).__init__()
        self.features = model.features[:-3]
        # GAP
        # 모델마다 변경 필요
        self.avg_pool = nn.AvgPool2d(8)

        # FC Layer
        self.classifier = nn.Linear(512 , nc)

    def forward(self, x):
        features = self.features(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        return output, features
```


```python
def model_train(mode):
    # scheduler.step()
    train_loss = []
    model.train()
    for iter,batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        if mode == 'CAM':
            pred, _ = model(x)
        else:
            pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        optimizer.step()      
        train_loss.append(np.sqrt(loss.item()))
        # if iter % 200 == 0:
        #     print("train loss", np.mean(train_loss))    

    return np.mean(train_loss)


def model_eval(epoch, mode) :
    model.eval()
    valid_loss = 0
    valid_accuracy = 0
    true = np.array([]); pred = np.array([])
    with torch.no_grad():
        for iter, batch in enumerate(valid_loader):
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            if mode == 'CAM':
                output, _ = model(x)
            else:
                output = model(x)
            loss = criterion(output, y)
            valid_loss += loss.item()/len(valid_loader)
            output = output.argmax(1)
            true = np.append(true, y.detach().to('cpu').numpy())
            pred = np.append(pred, output.detach().to('cpu').numpy())
    valid_accuracy = accuracy_score(true, pred)    
    return valid_loss, valid_accuracy
```


```python
def create_cam(model, input_x, real_img, label, img_size, classes):

    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())

    model.features[-1].register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (img_size, img_size)
        _, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

        
    logit,_ = model(input_x)
    logit = F.softmax(logit)
    idx = logit.max(1).indices
    probs = logit.max(1).values
    _, f, k, k = feature_blobs[0].shape
    fig, ax = plt.subplots(64, 1, figsize=(10, 350))
    for i in range(64):
        tmp_img = cv2.warpAffine(real_img[i], M_90, (w, h))
        txt = "True label : %d(%s), Predicted label : %d(%s), Probability : %.2f" % (label[i].item(), classes[label[i].item()], idx[i].item(), classes[idx[i].item()], probs[i].item())
        CAMs = returnCAM(feature_blobs[0][i].reshape(1,f,k,k), weight_softmax, [idx[i].item()])
        height, width, _ = tmp_img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_RAINBOW)
        ax[i].imshow(tmp_img.astype(np.uint))
        ax[i].imshow(heatmap.astype(np.uint), alpha = 0.4)
        ax[i].set_title(txt, fontsize = 10)
    plt.show()    
    feature_blobs.clear()
```

### STL10 데이터셋


```python
train_ds = datasets.STL10('/', split='train', download=True, transform=transform)
val_ds = datasets.STL10('/', split='test', download=True, transform=transform)
```

    Downloading http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz to /stl10_binary.tar.gz
    


      0%|          | 0/2640397119 [00:00<?, ?it/s]


    Extracting /stl10_binary.tar.gz to /
    Files already downloaded and verified
    


```python
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
```


```python
model = CAM_Model(vgg19(pretrained = True), 128,10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
```

    Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to /root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
    


      0%|          | 0.00/548M [00:00<?, ?B/s]



```python
'''
STL10
'''
best = 9999
for epoch in range(20):
    start=time.time()
    train_loss = model_train(mode = 'CAM')

    eval_loss, valid_accuracy = model_eval(epoch, 'CAM')

    if best > eval_loss:
        best = eval_loss
    print("{} epoch score,   eval loss : {}, eval accuracy : {},   time elapsed : {}".format(epoch, eval_loss, valid_accuracy,time.time() - start))
    print("best score : {}\n".format(best))
    print('===============================================')
```

<div class = long>
{% highlight python %}
    79it [00:32,  2.40it/s]
    

    0 epoch score,   eval loss : 0.270487082004547, eval accuracy : 0.911,   time elapsed : 50.25022840499878
    best score : 0.270487082004547
    
    ===============================================
    

    79it [00:26,  2.97it/s]
    

    1 epoch score,   eval loss : 0.23704791080951684, eval accuracy : 0.923375,   time elapsed : 44.259453535079956
    best score : 0.23704791080951684
    
    ===============================================
    

    79it [00:27,  2.91it/s]
    

    2 epoch score,   eval loss : 0.25278156316280376, eval accuracy : 0.915125,   time elapsed : 44.944968700408936
    best score : 0.23704791080951684
    
    ===============================================
    

    79it [00:27,  2.84it/s]
    

    3 epoch score,   eval loss : 0.29790895503759385, eval accuracy : 0.899375,   time elapsed : 45.9209988117218
    best score : 0.23704791080951684
    
    ===============================================
    

    79it [00:28,  2.80it/s]
    

    4 epoch score,   eval loss : 0.2720021572709084, eval accuracy : 0.91375,   time elapsed : 46.480770111083984
    best score : 0.23704791080951684
    
    ===============================================
    

    79it [00:28,  2.76it/s]
    

    5 epoch score,   eval loss : 0.34840211933851256, eval accuracy : 0.90275,   time elapsed : 46.9483323097229
    best score : 0.23704791080951684
    
    ===============================================
    

    79it [00:28,  2.75it/s]
    

    6 epoch score,   eval loss : 0.29445395085215564, eval accuracy : 0.918375,   time elapsed : 47.148518800735474
    best score : 0.23704791080951684
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    7 epoch score,   eval loss : 0.22611119088903073, eval accuracy : 0.941625,   time elapsed : 47.39880871772766
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:29,  2.72it/s]
    

    8 epoch score,   eval loss : 0.25590103073418147, eval accuracy : 0.9395,   time elapsed : 47.49596619606018
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    9 epoch score,   eval loss : 0.2660675615686922, eval accuracy : 0.939625,   time elapsed : 47.352192640304565
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:29,  2.72it/s]
    

    10 epoch score,   eval loss : 0.26890005410835144, eval accuracy : 0.941625,   time elapsed : 47.51648306846619
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:29,  2.72it/s]
    

    11 epoch score,   eval loss : 0.2676189986355602, eval accuracy : 0.941875,   time elapsed : 47.50736689567566
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:29,  2.72it/s]
    

    12 epoch score,   eval loss : 0.27550411625951515, eval accuracy : 0.9415,   time elapsed : 47.47885322570801
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    13 epoch score,   eval loss : 0.282252400215715, eval accuracy : 0.9415,   time elapsed : 47.43885016441345
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:29,  2.72it/s]
    

    14 epoch score,   eval loss : 0.28813641916960475, eval accuracy : 0.9415,   time elapsed : 47.4359176158905
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    15 epoch score,   eval loss : 0.29336506308987736, eval accuracy : 0.94125,   time elapsed : 47.27074193954468
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:29,  2.72it/s]
    

    16 epoch score,   eval loss : 0.2980969123505058, eval accuracy : 0.941125,   time elapsed : 47.42291736602783
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    17 epoch score,   eval loss : 0.3024260774869474, eval accuracy : 0.941125,   time elapsed : 47.4150664806366
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    18 epoch score,   eval loss : 0.3064234531372785, eval accuracy : 0.941125,   time elapsed : 47.37482404708862
    best score : 0.22611119088903073
    
    ===============================================
    

    79it [00:28,  2.73it/s]
    

    19 epoch score,   eval loss : 0.31012987919524304, eval accuracy : 0.941,   time elapsed : 47.36470413208008
    best score : 0.22611119088903073
    
    ===============================================
{% endhighlight %}
</div>    
<br><br>

```python
for iter, batch in enumerate(valid_loader):
    x = torch.tensor(batch[0], dtype=torch.float32, device=device)
    y = torch.tensor(batch[1], dtype=torch.long, device=device)
    output, _ = model(x)
    break

img = invTrans(x).transpose(3,1).to('cpu').detach().numpy()*255
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
create_cam(model, x, img, y, 128, classes)
```


<div class = long>
<img src = "/images/2022_09_19_Grad_CAM_Visual_Explanations_from_Deep_Networks_files/2022_09_19_Grad_CAM_Visual_Explanations_from_Deep_Networks_18_0.png">
</div>
    


## CAM 단점

CAM의 가장 큰 단점은 바로 Global Average Pooling layer가 반드시 필요하다는 점이다. GAP이 이미 포함되어 있는 GoogLeNet의 경우에는 문제가 없겠지만, 그렇지 않은 경우에는 마지막 convolutional layer 뒤에 GAP를 붙여서 다시 fine-tuning 해야 한다는 문제가 생기고, 약간의 성능 감소를 동반하는 문제가 있습니다. 또한, 같은 이유로 마지막 layer에 대해서만 CAM 결과를 얻을 수 있다.

또한 기존의 CAM은 여러가지 CNN model-families 중에 마지막에 FC Layer가 존재하는 Classification에만 적용 가능하다. 즉 Captioning, Visual Question Answering), 강화학습 등 다양한 모델에 적용이 불가능하다.


