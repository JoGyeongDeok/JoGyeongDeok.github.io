---
title: "Grad-CAM Visual Explanations from Deep Networks(2)"
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

# II. Grad-CAM

<img src = "https://miro.medium.com/max/1400/1*Ywow17bDVkNzBnIA8NAUxg.png" height = 300 width = 1200>

Convolutional layers에 남아있는 공간적인 정보는 fullly-connected layer에서 잃게 된다. 그래서 본 논문에서는 마지막 convolutional layer가 higher-level semantics와 detailed 공간정보의 적당한 타협점을 가진다고 생각했다.<br><br>

Grad-CAM은 CNN의 마지막 convolutional layer으로 가는 Gradient information를 사용하여 관심 있는 특정 결정에 대해 각 뉴런에 중요도 값을 할당한다. 즉, 마지막 convolution layer를 사용할 수 있다는 것이다.<br><br>

위의 사진에서 class-discriminative localization map Grad-CAM $L^c_{Grad-CAM}\in \mathbb{R}^{u \times v}$이다.($u\ :\ width,\ \ v\ :\ height,\ \ c\ :\ class$)
<br><br>
먼저 convolutional layer에 대한 feature map activations ($A^k$)를 가지는 class $c$에 대한 gradient score($y^c,\ softmax$함수에 넣기 전)를 계산한다. 즉 $\frac{\partial y^c}{\partial A^k}$를 계산한다. <br>
이후 Global-Average-pooling을 진행한 $\alpha^k$를 계산하는 동안, 정확한 계산은 Gradient가 전파되는 최종 Convolutional layers까지 가중치 행렬과 활성화 함수에 대한 행렬 곱에 해당한다. 따라서 따라서 이 가중치 $\alpha^k$는 A로부터 다운스트림의 심층 네트워크의 부분선형화를 나타내며 다음과 같이 나타낸다.

$$
\alpha^c_k\ =\ \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A^k}\\ L^C_{Grad-CAM}\ =\ ReLU(\sum_{k}\alpha^c_k A^k)
$$
<br><br>
위의 결과는 Convolutional feature maps와 같은 사이즈의 **coarse heatmap**이다.
또한 본 논문에서는 CAM의 일반화가 Grad-CAM라고 설명하고 있다.(이 내용은 생략한다.) 

## Grad-CAM 구현


```python
model = vgg19(pretrained = True).to(device)
model.classifier[-1] = nn.Linear(4096, 10, bias = True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
```


```python
def create_grad_cam(model, input_x, real_img, label, img_size, classes, target_layer):
    fmap_pool = {}
    grad_pool = {}
    handlers = []
    output_cam = []

    def save_fmaps(key):
        def forward_hook(module, input, output):
            fmap_pool[key] = output.detach()
        return forward_hook

    def save_grads(key):
        def backward_hook(module, grad_in, grad_out):
            grad_pool[key] = grad_out[0].detach()
        return backward_hook

    for name, module in model.named_modules():
        handlers.append(module.register_forward_hook(save_fmaps(name)))
        handlers.append(module.register_backward_hook(save_grads(name)))

    def _find( pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    output = model(x)
    output.backward(output, retain_graph = True)

    logit = model(input_x)
    logit = F.softmax(logit)
    idx = logit.max(1).indices
    probs = logit.max(1).values



    fmaps = _find(fmap_pool, target_layer)
    grads = _find(grad_pool, target_layer)
    weights = F.adaptive_avg_pool2d(grads, 1)

    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam = F.relu(gcam)
    gcam = F.interpolate(
        gcam, 128, mode="bilinear", align_corners=False
    )

    size_upsample = (img_size, img_size)
    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= gcam.max(dim=1, keepdim=True)[0]
    gcam = gcam.view(B, C, H, W)
    gcam = gcam.to('cpu').detach().numpy()
    gcam = np.uint8(255 * gcam).squeeze(1)

    fig, ax = plt.subplots(64, 1, figsize=(10, 350))
    for i in range(64):
        tmp_img = cv2.warpAffine(real_img[i], M_90, (w, h))
        txt = "True label : %d(%s), Predicted label : %d(%s), Probability : %.2f" % (label[i].item(), classes[label[i].item()], idx[i].item(), classes[idx[i].item()], probs[i].item())
        height, width, _ = tmp_img.shape
        heatmap = cv2.applyColorMap(cv2.resize(gcam[i], (width, height)), cv2.COLORMAP_RAINBOW)
        ax[i].imshow(tmp_img.astype(np.uint))
        ax[i].imshow(heatmap.astype(np.uint), alpha = 0.4)
        ax[i].set_title(txt, fontsize = 10)
    plt.show()    

```


```python
'''
STL10
'''
best = 9999
for epoch in range(20):
    start=time.time()
    train_loss = model_train(mode = 'Grad')

    eval_loss, valid_accuracy = model_eval(epoch, 'Grad')

    if best > eval_loss:
        best = eval_loss
    print("{} epoch score,   eval loss : {}, eval accuracy : {},   time elapsed : {}".format(epoch, eval_loss, valid_accuracy,time.time() - start))
    print("best score : {}\n".format(best))
    print('===============================================')
```

<div class = long>
{% highlight python %}
    79it [00:30,  2.60it/s]
    

    0 epoch score,   eval loss : 0.18785101974010468, eval accuracy : 0.939875,   time elapsed : 49.25390839576721
    best score : 0.18785101974010468
    
    ===============================================
    

    79it [00:31,  2.53it/s]
    

    1 epoch score,   eval loss : 0.19060212589800363, eval accuracy : 0.940875,   time elapsed : 50.442219734191895
    best score : 0.18785101974010468
    
    ===============================================
    

    79it [00:31,  2.49it/s]
    

    2 epoch score,   eval loss : 0.2004742234125734, eval accuracy : 0.940625,   time elapsed : 51.00742697715759
    best score : 0.18785101974010468
    
    ===============================================
    

    79it [00:32,  2.46it/s]
    

    3 epoch score,   eval loss : 0.17100459837913518, eval accuracy : 0.94875,   time elapsed : 51.45760631561279
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    4 epoch score,   eval loss : 0.21955611729621893, eval accuracy : 0.9385,   time elapsed : 52.03212809562683
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    5 epoch score,   eval loss : 0.2255802784785628, eval accuracy : 0.9405,   time elapsed : 52.10357117652893
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    6 epoch score,   eval loss : 0.21616217952780425, eval accuracy : 0.9485,   time elapsed : 52.01284456253052
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    7 epoch score,   eval loss : 0.19647681984677917, eval accuracy : 0.949875,   time elapsed : 52.05117988586426
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    8 epoch score,   eval loss : 0.21037958858162167, eval accuracy : 0.945375,   time elapsed : 52.08625030517578
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    9 epoch score,   eval loss : 0.19549667561799286, eval accuracy : 0.954125,   time elapsed : 51.9943368434906
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    10 epoch score,   eval loss : 0.1909742392972112, eval accuracy : 0.9515,   time elapsed : 52.049251079559326
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    11 epoch score,   eval loss : 0.19697506992146385, eval accuracy : 0.954375,   time elapsed : 51.959964990615845
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    12 epoch score,   eval loss : 0.20169063849747176, eval accuracy : 0.9545,   time elapsed : 52.062161684036255
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    13 epoch score,   eval loss : 0.21628177276859054, eval accuracy : 0.9515,   time elapsed : 52.05464744567871
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.42it/s]
    

    14 epoch score,   eval loss : 0.20970297218346975, eval accuracy : 0.95325,   time elapsed : 52.072774171829224
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    15 epoch score,   eval loss : 0.21972980161092695, eval accuracy : 0.952875,   time elapsed : 51.99079871177673
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    16 epoch score,   eval loss : 0.21851732936629562, eval accuracy : 0.954125,   time elapsed : 52.051114082336426
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    17 epoch score,   eval loss : 0.22268633013498038, eval accuracy : 0.952875,   time elapsed : 52.0331072807312
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    18 epoch score,   eval loss : 0.22275295946607365, eval accuracy : 0.95375,   time elapsed : 52.00172781944275
    best score : 0.17100459837913518
    
    ===============================================
    

    79it [00:32,  2.43it/s]
    

    19 epoch score,   eval loss : 0.22606607300171178, eval accuracy : 0.954375,   time elapsed : 52.00130271911621
    best score : 0.17100459837913518
    
    ===============================================
{% endhighlight %}
</div>
<br>    


```python
for iter, batch in enumerate(valid_loader):
    x = torch.tensor(batch[0], dtype=torch.float32, device=device)
    y = torch.tensor(batch[1], dtype=torch.long, device=device)
    output = model(x)
    break

img = invTrans(x).transpose(3,1).to('cpu').detach().numpy()*255
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
cam = create_grad_cam(model, x, img, y, 128, classes,'features')
```

<div class = long>
<img src = "/images/2022_09_19_Grad_CAM_Visual_Explanations_from_Deep_Networks_files/2022_09_19_Grad_CAM_Visual_Explanations_from_Deep_Networks_28_0.png">
</div>
<br>


# Reference
[1] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning deep features for discriminative localization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2921-2929).

[2] https://www.secmem.org/blog/2020/01/17/gradcam/

[3] https://github.com/KangBK0120/CAM/blob/master/create_cam.py

[4] https://github.com/kazuto1011/grad-cam-pytorch/blob/master/grad_cam.py

[5] https://hongl.tistory.com/157

