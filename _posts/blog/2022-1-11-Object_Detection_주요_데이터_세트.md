---
title: "Object Detection 주요 데이터 세트"
category: DeepLearning
tags: [Computer Vision, Object Detection, Deep Learning]
comments: true
date : 2022-01-11
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

 - **많은 Detection과 Segmentation 딥러닝 패키지가 Dataset들을 기반으로 Pretrained 되어 배포되어 있다.**
- PASCAL VOC (XML Format)
> 20개의 오브젝트 카테고리
- MS COCO (json Format)
> 80개의 오브젝트 카테고리
- Google open Images (csv Format)
> 600개의 오브젝트 카테고리

<br>

# PASCAL VOC


```python
# pascal voc 2012 데이터를 다운로드 후 디렉토리에 압축 해제
!mkdir '/content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC'
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf VOCtrainval_11-May-2012.tar -C '/content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC'
```


```python
! ls '/content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC/VOCdevkit/VOC2012'
! ls '/content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC/VOCdevkit/VOC2012/JPEGImages' | head -n 5
path = '/content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC/VOCdevkit/VOC2012/'
```

    Annotations  ImageSets	JPEGImages  SegmentationClass  SegmentationObject
    2007_000027.jpg
    2007_000032.jpg
    2007_000033.jpg
    2007_000039.jpg
    2007_000042.jpg
    

***JPEGImages 디렉토리에 있는 임의의 이미지 보기***


```python
import cv2 
import matplotlib.pyplot as plt
import os
%matplotlib inline

img = cv2.imread(path + 'JPEGImages/2007_000032.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape : ', img.shape)

plt.figure(figsize = (8,8))
plt.imshow(img_rgb)
plt.show()
```

    img shape :  (281, 500, 3)
    


    
![png](/images/2022-1-11-Object_Detection_%EC%A3%BC%EC%9A%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_files/2022-1-11-Object_Detection_%EC%A3%BC%EC%9A%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_6_1.png)
    


***Annotations 디렉토리에 있는 임의의 annotation 파일 보기***


```python
!cat '/content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC/VOCdevkit/VOC2012/Annotations/2007_000032.xml'
```

    <annotation>
    	<folder>VOC2012</folder>
    	<filename>2007_000032.jpg</filename>
    	<source>
    		<database>The VOC2007 Database</database>
    		<annotation>PASCAL VOC2007</annotation>
    		<image>flickr</image>
    	</source>
    	<size>
    		<width>500</width>
    		<height>281</height>
    		<depth>3</depth>
    	</size>
    	<segmented>1</segmented>
    	<object>
    		<name>aeroplane</name>
    		<pose>Frontal</pose>
    		<truncated>0</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>104</xmin>
    			<ymin>78</ymin>
    			<xmax>375</xmax>
    			<ymax>183</ymax>
    		</bndbox>
    	</object>
    	<object>
    		<name>aeroplane</name>
    		<pose>Left</pose>
    		<truncated>0</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>133</xmin>
    			<ymin>88</ymin>
    			<xmax>197</xmax>
    			<ymax>123</ymax>
    		</bndbox>
    	</object>
    	<object>
    		<name>person</name>
    		<pose>Rear</pose>
    		<truncated>0</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>195</xmin>
    			<ymin>180</ymin>
    			<xmax>213</xmax>
    			<ymax>229</ymax>
    		</bndbox>
    	</object>
    	<object>
    		<name>person</name>
    		<pose>Rear</pose>
    		<truncated>0</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>26</xmin>
    			<ymin>189</ymin>
    			<xmax>44</xmax>
    			<ymax>238</ymax>
    		</bndbox>
    	</object>
    </annotation>
    

***Segmentation png파일 확인***


```python
img = cv2. imread(path + "SegmentationObject/2007_000032.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape : ', img.shape)

plt.figure(figsize = (8,8))
plt.imshow(img_rgb)
plt.show()
```

    img shape :  (281, 500, 3)
    


    
![png](/images/2022-1-11-Object_Detection_%EC%A3%BC%EC%9A%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_files/2022-1-11-Object_Detection_%EC%A3%BC%EC%9A%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_10_1.png)
    


***Annotation xml 파일에 있는 요소들을 파싱하여 접근하기***
- Elememt Tree를 이용하여 XML 파싱


```python
import os
import random
import xml.etree.ElementTree as ET

xml_files = os.listdir(path + "Annotations")
print(xml_files[:5]); print(len(xml_files))
```

    ['2007_000027.xml', '2007_000032.xml', '2007_000033.xml', '2007_000039.xml', '2007_000042.xml']
    17125
    


```python
xml_file = path + "Annotations/2007_000032.xml"

# XML 파일을 Parsing하여 Element 생성
tree = ET.parse(xml_file)
root = tree.getroot()

# image 관련 정보는 root의 자식으로 존재
image_name = root.find('filename').text
full_image_name = path + "JPEGImages/" + image_name
image_size = root.find('size')
image_width = int(image_size.find('width').text)
image_height = int(image_size.find('height').text)

# 파일 내에 있는 모든 object Element를 찾음
object_list = []

# 이미지 
img = cv2.imread(full_image_name)
draw_img = img.copy()
green_color = (0, 255, 0)
red_color = (0, 0, 255)

for obj in root.findall('object'):
  # object element의 자식 element에서 bndbox를 찾음
  xmlbox = obj.find('bndbox')
  # bndbox element의 자식 element에서 xmin, ymin, xmax, ymax를 찾고 이의 값(text)를 추출
  x1 = int(xmlbox.find('xmin').text)    #left
  y1 = int(xmlbox.find('ymin').text)    #top
  x2 = int(xmlbox.find('xmax').text)    #right
  y2 = int(xmlbox.find('ymax').text)    #bottom

  bndbox_pos = (x1, y1, x2, y2)
  class_name = obj.find('name').text
  object_dict = {'class_name' : class_name, 'bndbox_pos' : bndbox_pos}
  object_list.append(object_dict)

  cv2.rectangle(draw_img, (x1, y1), (x2, y2), color = green_color, thickness = 1)
  cv2.putText(draw_img, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, thickness=1)


img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (10, 10))
plt.imshow(img_rgb)


print('full_image_name : ', full_image_name, '\nimage_size : ', (image_width, image_height))
for object in object_list:
  print(object)

```

    full_image_name :  /content/drive/MyDrive/딥러닝/Computer Vision/Image/PASCAL VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg 
    image_size :  (500, 281)
    {'class_name': 'aeroplane', 'bndbox_pos': (104, 78, 375, 183)}
    {'class_name': 'aeroplane', 'bndbox_pos': (133, 88, 197, 123)}
    {'class_name': 'person', 'bndbox_pos': (195, 180, 213, 229)}
    {'class_name': 'person', 'bndbox_pos': (26, 189, 44, 238)}
    


    
![png](/images/2022-1-11-Object_Detection_%EC%A3%BC%EC%9A%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_files/2022-1-11-Object_Detection_%EC%A3%BC%EC%9A%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_13_1.png)
    


