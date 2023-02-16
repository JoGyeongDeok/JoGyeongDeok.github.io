---
title: "AI competition to diagnose crop diseases due to changes in the agricultural environment"
category: DeepLearning
tags: [Computer Vision, Object Detection, Deep Learning, Dacon, Yolo]
comments: true
date : 2022-01-17
categories: 
  - blog
excerpt: CV
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
use_math: true
---
<br>

# 1. Library & Data Load & 함수정의


```python
from google.colab import drive
drive.mount('/content/drive')
```


```python
!pip install ray                                    #병렬처리 
!git clone https://github.com/ultralytics/yolov5    #Yolo
!cd yolov5; pip install -qr requirements.txt
```


```python
import sys
sys.path
```


```python
cp -r '/content/yolov5/utils' '/content'
cp -r '/content/yolov5/models' '/content'
```


```python
path= '/content/drive/MyDrive/DACON/Data/AI competition to diagnose crop diseases due to changes in the agricultural environment'
```


```python
from time import sleep
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os
import json 
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import yaml
import random
from google.colab.patches import cv2_imshow

import ray
import datetime
import time

```


```python
#os.mkdir(path)
# # zipfile.ZipFile(path+'.zip','r').extractall(path+'/')
# # zipfile.ZipFile(path+'/train.zip','r').extractall(path+'/')
# # zipfile.ZipFile(path+'/test.zip','r').extractall(path+'/')

#os.mkdir(path+'/images')
#os.mkdir(path+'/labels')

#os.mkdir(path+'/images/train')
#os.mkdir(path+'/images/valid')
#os.mkdir(path+'/images/test')

#os.mkdir(path+'/labels/train')
#os.mkdir(path+'/labels/valid')
#os.mkdir(path+'/labels/test')

# os.mkdir(path + '/ultra_workdir')
# os.makedirs(path + '/CNN_model')
```


```python
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print('Using PyTorch version : ', torch.__version__, 'Device : ', device)
```
<br>

# 2. Object Deteion 수행

## Yolo 형식 맞춰서 이미지, annotation 파일 저장


```python
test_index = []
for _ in range(10):
  if len(test_index) != 0 :
    break

  train_index = sorted(glob(path+"/train/*"))
  test_index = sorted(glob(path+"/test/*"))
  
  for num in range(len(train_index)):
    train_index[num] = train_index[num] + train_index[num][len(path + '/train'):]
    
  for num in range(len(test_index)):
    test_index[num] = test_index[num] + test_index[num][len(path + '/test'):]
    
```


```python
#51906
len(test_index)
```




    51906




```python
#5767
len(train_index)
```




    5767




```python
# 1개의 voc xml 파일을 Yolo 포맷용 txt 파일로 변경하는 함수 
def json_to_txt(input_xml_file, output_txt_file):
  # 원본 이미지의 너비와 높이 추출. 
  with open(input_xml_file, 'r', encoding='UTF-8') as json_file:
    json_file = json.load(json_file)
    img_width = int(json_file['description']['width'])
    img_height = int(json_file['description']['height'])
    x1 = int(json_file['annotations']['bbox'][0]['x'])
    y1 = int(json_file['annotations']['bbox'][0]['y'])
    x2 = int(json_file['annotations']['bbox'][0]['x']) + int(json_file['annotations']['bbox'][0]['w'])
    y2 = int(json_file['annotations']['bbox'][0]['y']) + int(json_file['annotations']['bbox'][0]['h'])
    object_name = json_file['annotations']['crop']

  # object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환하는 convert_yolo_coord()함수 호출. 
  cx_norm, cy_norm, w_norm, h_norm = convert_yolo_coord(object_name, img_width, img_height, x1, y1, x2, y2)
  # 변환된 yolo 좌표를 object 별로 출력 text 파일에 write
  value_str = ('{0} {1} {2} {3} {4}').format(object_name-1, cx_norm, cy_norm, w_norm, h_norm)
  with open(output_txt_file, 'w') as output_fpointer:
    output_fpointer.write(value_str+'\n')


# object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환
def convert_yolo_coord(object_name, img_width, img_height, x1, y1, x2, y2):
  # 중심 좌표와 너비, 높이 계산. 
  center_x = (x1 + x2)/2
  center_y = (y1 + y2)/2
  width = x2 - x1
  height = y2 - y1
  # 원본 이미지 기준으로 중심 좌표와 너비 높이를 0-1 사이 값으로 scaling
  center_x_norm = center_x / img_width
  center_y_norm = center_y / img_height
  width_norm = width / img_width
  height_norm = height / img_height

  return round(center_x_norm, 7), round(center_y_norm, 7), round(width_norm, 7), round(height_norm, 7)
```


```python
shutil.mov
```


```python
def make_yolo_anno_file(index, tgt_images_dir, tgt_labels_dir, image, label):
  for row in tqdm(index):
    src_image_path = row + '.jpg'
    src_label_path = row + '.json'
    # yolo format으로 annotation할 txt 파일의 절대 경로명을 지정. 
      
    if image == True : 
      # image의 경우 target images 디렉토리로 단순 copy
      shutil.copy(src_image_path, tgt_images_dir)     

    if label == True : 
      target_label_path = tgt_labels_dir + row[-5:]+'.txt'
      # annotation의 경우 json 파일을 target labels 디렉토리에 Ultralytics Yolo format으로 변환하여  만듬
      json_to_txt(src_label_path, target_label_path)
```


```python
random.seed(42)
valid_index = random.sample(train_index,int(len(train_index)*0.1))
```


```python
# # # train용 images와 labels annotation 생성. 
# make_yolo_anno_file(train_index, path+'/images/train/', path+'/labels/train/', image = True, label = True)
make_yolo_anno_file(valid_index, path+'/images/valid/', path+'/labels/valid/', image = True, label = True)
```


```python
# yaml파일 작성
#with open(path+"/crop.yaml","w") as f:
#  f.write(f'train: {path}/images/train/\n')
#  f.write(f'val: {path}/images/valid/\n')  
#  f.write(f'test: {path}/images/test/\n')
#  f.write(f'nc: 6\n')
#  f.write(f'names: [ 0, 1, 2, 3, 4, 5 ]')
```

## YOLO 모델 Train


```python
###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. 
# !cd /content/yolov5; python train.py --img 640 --batch 16 --epochs 30 --data '/content/drive/MyDrive/DACON/Data/AI competition to diagnose crop diseases due to changes in the agricultural environment/crop.yaml' --weights yolov5x.pt --project='/content/drive/MyDrive/DACON/Data/AI competition to diagnose crop diseases due to changes in the agricultural environment/ultra_workdir/Detectionmodel1' \
#                                      --exist-ok
```

## Yolo Detect 및 Cut


### Yolo library


```python
!pip install timm
```

    Requirement already satisfied: timm in /usr/local/lib/python3.7/dist-packages (0.4.12)
    Requirement already satisfied: torch>=1.4 in /usr/local/lib/python3.7/dist-packages (from timm) (1.10.0+cu111)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.7/dist-packages (from timm) (0.11.1+cu111)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch>=1.4->timm) (3.10.0.2)
    Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in /usr/local/lib/python3.7/dist-packages (from torchvision->timm) (7.1.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torchvision->timm) (1.19.5)
    


```python
import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import json
import cv2
import matplotlib.pyplot as plt

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import time



device = torch.device('cuda')
def accuracy_function(real, pred):    
    score = f1_score(real, pred, average='macro')
    return score

def model_save(model, score,  path):
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'score': score
    }, path)

```


```python
crops = list(range(len(train_index)))
diseases = list(range(len(train_index)))
risks = list(range(len(train_index)))
labels = list(range(len(train_index)))
bboxs = list(range(len(train_index)))
for i in tqdm(range(len(train_index))):
    with open(train_index[i]+'.json', 'r') as f:
        sample = json.load(f)
    crop = sample['annotations']['crop']
    disease = sample['annotations']['disease']
    risk = sample['annotations']['risk']
    label=f"{crop}_{disease}_{risk}"
    bbox = sample['annotations']['bbox']
    crops[i] = crop
    diseases[i] = disease
    risks[i] = risk
    labels[i] = label
    bboxs[i] = bbox
```

    100%|██████████| 5767/5767 [00:05<00:00, 1125.95it/s]
    


```python
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

def test_cut(source, weights, data, 
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        augment=False,  # augmented inference
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False # use OpenCV DNN for ONNX inference
        ):  
    imgs = []
    # Load model

    model = DetectMultiBackend(weights, device = device, dnn = dnn, data = data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for i in tqdm(range(len(source))):
        im = np.ascontiguousarray(source[i].transpose((2,0,1))[::-1])
        im0s = source[i]
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2


        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        if det.size()[0] > 0:
          temp_img = im0s[int(det[0][1]):int(det[0][3]), int(det[0][0]):int(det[0][2])]
        else:
          temp_img = im0s

        temp_img = cv2.resize(temp_img, (384, 512))
        
        imgs.append(temp_img)
    return imgs
```

### Train Cut


```python
# Ray Task    
ray.shutdown()
ray.init()
@ray.remote
def train_img_load(path,bbox):
    img = cv2.imread(path)[:,:,::-1]
    img = img[int(bbox[0]['y']):(int(bbox[0]['y'])+int(bbox[0]['h'])), int(bbox[0]['x']):(int(bbox[0]['x'])+int(bbox[0]['w']))]
    img = cv2.resize(img, (384, 512))
    return img
```


```python
start = time.time()
result = []
for k,pa in tqdm(enumerate(train_index)):
  result.append(train_img_load.remote(ray.put(pa+".jpg"), ray.put(bboxs[k])))
train_img = ray.get(result)
ray.shutdown()

end = time.time()
print('소요 시간 : ', end - start)
```

    5767it [00:10, 535.16it/s]
    

    [2m[36m(train_img_load pid=24920)[0m 
    소요 시간 :  40.85077214241028
    

### Test Cut


```python
ray.shutdown()
ray.init()
@ray.remote
def test_img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (384, 512))
    return img
```


```python
temp_test = []
```


```python
for i in range(10):
  start = time.time()

  ray.shutdown()
  ray.init()
  result = []
  if i < 9:
    for pa in tqdm(test_index[i*5000:(i+1)*5000]):
      result.append(test_img_load.remote(ray.put(pa+".jpg")))
  else:
    for pa in tqdm(test_index[i*5000:]):
      result.append(test_img_load.remote(ray.put(pa+".jpg")))
  temp_test.append(ray.get(result)) 
  end = time.time()
  print(i,' 소요 시간 : ', end - start)

ray.shutdown()
```

    2022-01-17 01:28:53,386	WARNING services.py:1826 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 6541639680 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=7.76gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
    100%|██████████| 6906/6906 [00:07<00:00, 975.59it/s]
    

    [2m[36m(test_img_load pid=28134)[0m 
    9  소요 시간 :  63.91328740119934
    


```python
len(temp_test)
```




    10

<br>

# 3. CNN


```python
class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        
        if self.mode=='train':
            augmentation = random.randint(0,2)
            if augmentation==1:
                img = img[::-1].copy()
            elif augmentation==2:
                img = img[:,::-1].copy()
        img = transforms.ToTensor()(img)
        if self.mode=='test':
            return img
        
        label = self.labels[idx]
        return img,label
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=25)
        
    def forward(self, x):
        x = self.model(x)
        return x    
```


```python
folds = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, valid_idx in kf.split(train_img):
    folds.append((train_idx, valid_idx))
fold=0
train_idx, valid_idx = folds[fold]

batch_size = 16
epochs = 30


train_dataset = Custom_dataset(np.array(train_img)[train_idx], np.array(labels)[train_idx], mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=8)

        
valid_dataset = Custom_dataset(np.array(train_img)[valid_idx], np.array(labels)[valid_idx], mode='valid')
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8)
```


```python
model = Network().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() 


best=0
for epoch in range(epochs):
    start=time.time()
    train_loss = 0
    train_pred=[]
    train_y=[]
    model.train()
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()
        
    
    train_f1 = accuracy_function(train_y, train_pred)
    
    model.eval()
    valid_loss = 0
    valid_pred=[]
    valid_y=[]
    with torch.no_grad():
        for batch in (valid_loader):
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                pred = model(x)
            loss = criterion(pred, y)
            valid_loss += loss.item()/len(valid_loader)
            valid_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            valid_y += y.detach().cpu().numpy().tolist()
        valid_f1 = accuracy_function(valid_y, valid_pred)
    if valid_f1>=best:
        best=valid_f1
        model_save(model, valid_f1, path + f'/CNN_model/eff-b0.pth')
    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    print(f'VALID    loss : {valid_loss:.5f}    f1 : {valid_f1:.5f}    best : {best:.5f}')
```


```python
loaded_model = Network()
loaded_model.cuda()
loaded_model.load_state_dict(torch.load(path + '/CNN_model/eff-b0.pth')['model'])
loaded_model.eval
```


```python
len(temp_test)
```




    10




```python
start = time.time()
test_pred = []

for i in range(len(temp_test)):
  print(i + 1 , "번째 Test 리스트")
  test_img = test_cut(source = temp_test[i], weights = path + '/ultra_workdir/Detectionmodel1/exp/weights/best.pt', data = path + '/crop.yaml')


  #test_Dataset
  temp_label = []
  for i in range(len(test_img)):
    temp_label.append('5')

  test_dataset = Custom_dataset(np.array(test_img),np.array(temp_label), mode='valid')
  test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8)

  #test pred
  with torch.no_grad():
      for batch in (test_loader):
          x = torch.tensor(batch[0], dtype=torch.float32, device=device)
          with torch.cuda.amp.autocast():
              pred = loaded_model(x)
          test_pred += pred.argmax(1).detach().cpu().numpy().tolist()

end = time.time()  
print('소요 시간 : ', end - start)
```

    1 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:15<00:00, 37.01it/s]
    

    2 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:20<00:00, 35.52it/s]
    

    3 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [03:07<00:00, 26.60it/s]
    

    4 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [03:07<00:00, 26.60it/s]
    

    5 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:51<00:00, 29.13it/s]
    

    6 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:36<00:00, 31.91it/s]
    

    7 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:55<00:00, 28.48it/s]
    

    8 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:40<00:00, 31.08it/s]
    

    9 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 5000/5000 [02:46<00:00, 30.09it/s]
    

    10 번째 Test 리스트
    

    Fusing layers... 
    Model Summary: 444 layers, 86207059 parameters, 0 gradients, 204.1 GFLOPs
    

    WARNING: --img-size (640, 640) must be multiple of max stride 32, updating to [640, 640]
    

    100%|██████████| 6906/6906 [03:22<00:00, 34.11it/s]
    

    소요 시간 :  2237.8540785312653
    


```python
label_description = [
                      "1_00_0",
                      "2_00_0",
                      "2_a5_2",
                      "3_00_0",
                      "3_a9_1",
                      "3_a9_2",
                      "3_a9_3",
                      "3_b3_1",
                      "3_b6_1",
                      "3_b7_1",
                      "3_b8_1",
                      "4_00_0",
                      "5_00_0",
                      "5_a7_2",
                      "5_b6_1",
                      "5_b7_1",
                      "5_b8_1",
                      "6_00_0",
                      "6_a11_1",
                      "6_a11_2",
                      "6_a12_1",
                      "6_a12_2",
                      "6_b4_1",
                      "6_b4_3",
                      "6_b5_1"
                      ]
```


```python
len(test_pred)
```




    51906




```python
final_pred = []
for i in range(len(test_pred)):
  final_pred.append(label_description[test_pred[i]])
```


```python
submission = pd.read_csv(path + '/sample_submission.csv')
submission['label'] = final_pred
submission
```





  <div id="df-24b61699-da1c-4994-b32a-13d6cd6eb47c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000</td>
      <td>6_00_0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10001</td>
      <td>5_b6_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10002</td>
      <td>4_00_0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10003</td>
      <td>3_00_0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10004</td>
      <td>3_b8_1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51901</th>
      <td>67673</td>
      <td>4_00_0</td>
    </tr>
    <tr>
      <th>51902</th>
      <td>67674</td>
      <td>3_b7_1</td>
    </tr>
    <tr>
      <th>51903</th>
      <td>67675</td>
      <td>6_00_0</td>
    </tr>
    <tr>
      <th>51904</th>
      <td>67676</td>
      <td>2_a5_2</td>
    </tr>
    <tr>
      <th>51905</th>
      <td>67677</td>
      <td>6_00_0</td>
    </tr>
  </tbody>
</table>
<p>51906 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-24b61699-da1c-4994-b32a-13d6cd6eb47c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-24b61699-da1c-4994-b32a-13d6cd6eb47c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-24b61699-da1c-4994-b32a-13d6cd6eb47c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
submission.to_csv(path + '/submission/submission.csv', index=False)
```
