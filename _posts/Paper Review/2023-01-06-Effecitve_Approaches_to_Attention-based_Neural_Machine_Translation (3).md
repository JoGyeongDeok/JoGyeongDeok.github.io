---
title: "Effecitve Approaches to Attention-based Neural Machine Translation (3)"
tags: [Pytorch, Deep Learning, Attention]
comments: true
excerpt: Attention
date : 2023-01-06
categories: 
  - PaperReview
layout: jupyter
search: true
# 목차
toc: true  
toc_sticky: true 
category : NLP
---
# IV. Implement

## 3. Neural Machine Translation 


```python
vocab_inp_size = len(input_lang.word2index)
vocab_out_size = len(output_lang.word2index)
```


```python
# https://blog.naver.com/wooy0ng/222904552997

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

    
# EOS token이 나오면 loss 계산 중지
def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)
```


```python
class CustomDataset(Dataset):
    def __init__(self, input, target = None):
        self.input = input
        self.target = target
        # TODO: convert this into torch code is possible
        self.length = [ torch.sum(x!=0) for x in input]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x_len = self.length[idx]
        if self.target is not None:
            input_tensor, target_tensor = self.input[idx], self.target[idx]
            return input_tensor, target_tensor, x_len
        else:
            input_tensor = self.input[idx]
            return input_tensor, x_len
```


```python
train_input, valid_input, train_target, valid_target = train_test_split(input_tensor, target_tensor, test_size = 0.2, random_state = 50)

batch_size = 64
train_buffer_size = train_input.shape[0]
train_n_batch = train_buffer_size//batch_size

valid_buffer_size = valid_input.shape[0]
valid_n_batch = valid_buffer_size//batch_size

train_dataset = CustomDataset(train_input, train_target)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True,)
vali_dataset = CustomDataset(valid_input, valid_target)
vali_loader = DataLoader(vali_dataset, batch_size = batch_size, shuffle=False, drop_last=True,)
```


```python
def train(encoder, decoder, optimizer, data_loader):
    
    encoder.train()
    decoder.train()
    
    total_loss = 0
    for (batch, (inp, targ, inp_len)) in enumerate(data_loader):
        loss = 0
        xs, ys, lens = sort_batch(inp, targ, inp_len)
        enc_output, enc_hidden = encoder(xs.to(device), lens.to('cpu'), inp.size(0))
        dec_hidden = enc_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_input = torch.tensor([[SOS_token]]*inp.size(0), device=device)
        if use_teacher_forcing:
            for t in range(0, targ.size(1)):
                decoder_output, dec_hidden, _ = decoder(
                    decoder_input, dec_hidden, enc_output)
                loss += loss_function(ys[:,t], decoder_output)
                decoder_input = ys[:, t].unsqueeze(1)  # Teacher forcing. 다음 값 예측을 위해 Ground Truth 값을 사용
        else:
            for t in range(0, targ.size(1)):
                decoder_output, dec_hidden, _ = decoder(
                    decoder_input, dec_hidden, enc_output)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi
                loss += loss_function(ys[:,t], decoder_output)
        
        batch_loss = (loss / int(ys.size(1)))
        total_loss += batch_loss
        
            
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        if batch % 100 == 0:
            print('Batch {} Loss {:.4f}'.format(batch, batch_loss.detach().item()))
        
    final_loss = (total_loss / train_n_batch).detach().item()
    ### TODO: Save checkpoint for model
    print('Train Loss {:.4f}'.format(final_loss))
    return final_loss
```


```python
def validation(encoder, decoder,  optimizer, data_loader):
   
    encoder.eval()
    decoder.eval()


    total_loss = 0
    with torch.no_grad():
        for (batch, (inp, targ, inp_len)) in enumerate(data_loader):
            loss = 0
            xs, ys, lens = sort_batch(inp, targ, inp_len)
            enc_output, enc_hidden = encoder(xs.to(device), lens.to('cpu'), inp.size(0))
            dec_hidden = enc_hidden
            
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            decoder_input = torch.tensor([[SOS_token]]*inp.size(0), device=device)
            if use_teacher_forcing:
                for t in range(0, targ.size(1)):
                    decoder_output, dec_hidden, _ = decoder(
                        decoder_input, dec_hidden, enc_output)
                    loss += loss_function(ys[:,t], decoder_output)
                    decoder_input = ys[:, t].unsqueeze(1)  # Teacher forcing. 다음 값 예측을 위해 Ground Truth 값을 사용
            else:
                for t in range(0, targ.size(1)):
                    decoder_output, dec_hidden, _ = decoder(
                        decoder_input, dec_hidden, enc_output)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi
                    loss += loss_function(ys[:,t], decoder_output)
            
            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss

    final_loss = (total_loss / valid_n_batch).detach().item()
    ### TODO: Save checkpoint for model
    print('Validation Loss {:.4f}'.format(final_loss))
    return final_loss

```

### 3.1 NMT


```python
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, input, inp_len, bt_size):
        embedded = self.embedding(input)
        
        # x transformed = max_len X batch_size X embedding_dim
        # x = x.permute(1,0,2)
        output = pack_padded_sequence(embedded, inp_len) # unpad
        self.hidden = self.initHidden(bt_size)
        
        # pad the sequence to the max length in the batch
        

        output, self.hidden = self.gru(output, self.hidden)

        output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

    def initHidden(self, bt_size):
        return torch.zeros(1, bt_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, enc_output):
        output = self.embedding(input)
        output = F.relu(output)
        # hidden = hidden.permute(1,0,2)
        output, self.hidden = self.gru(output, hidden)
        output = self.softmax(self.fc(output.view(-1, output.size(2))))
        return output, self.hidden, None

    def initHidden(self, bt_size):
        return torch.zeros(1, bt_size, self.hidden_size, device=device)
```


```python
hidden_size = 1024
embedding_dim = 256

encoder = EncoderRNN(input_lang.n_words, embedding_dim, hidden_size).to(device)
decoder = DecoderRNN(output_lang.n_words,embedding_dim, hidden_size).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                       lr=0.001)

criterion = nn.CrossEntropyLoss()

```


```python
teacher_forcing_ratio = 0.5
epochs = 30
train_loss_list = []
valid_loss_list = []
best_score = 9999
for epoch in range(epochs):
    print('==============={}============'.format(epoch+1))

    start = time.time()
    train_loss = train(encoder, decoder, optimizer, data_loader = train_loader)
    valid_loss = validation(encoder, decoder, optimizer,  data_loader = vali_loader)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    if best_score > train_loss:
        best_score = train_loss
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Train Best Score :', best_score)
```

    ===============1============
    Batch 0 Loss 3.6871
    Batch 100 Loss 1.2280
    Batch 200 Loss 1.2461
    Batch 300 Loss 1.1257
    Batch 400 Loss 0.8549
    Batch 500 Loss 1.0651
    Train Loss 1.1266
    Validation Loss 0.8206
    Time taken for 1 epoch 38.370532274246216 sec
    
    Train Best Score : 1.1266220808029175
    ===============2============
    Batch 0 Loss 0.5838
    Batch 100 Loss 0.7077
    Batch 200 Loss 0.5201
    Batch 300 Loss 0.7356
    Batch 400 Loss 0.4910
    Batch 500 Loss 0.4301
    Train Loss 0.6365
    Validation Loss 0.6173
    Time taken for 1 epoch 38.44290471076965 sec
    
    Train Best Score : 0.6365498900413513
    ===============3============
    Batch 0 Loss 0.2888
    Batch 100 Loss 0.5145
    Batch 200 Loss 0.5279
    Batch 300 Loss 0.3718
    Batch 400 Loss 0.3236
    Batch 500 Loss 0.4649
    Train Loss 0.3925
    Validation Loss 0.5176
    Time taken for 1 epoch 37.71321439743042 sec
    
    Train Best Score : 0.3925304412841797
    ===============4============
    Batch 0 Loss 0.3352
    Batch 100 Loss 0.1839
    Batch 200 Loss 0.1603
    Batch 300 Loss 0.1998
    Batch 400 Loss 0.2191
    Batch 500 Loss 0.3506
    Train Loss 0.2479
    Validation Loss 0.5277
    Time taken for 1 epoch 38.07337760925293 sec
    
    Train Best Score : 0.24785540997982025
    ===============5============
    Batch 0 Loss 0.1768
    Batch 100 Loss 0.1650
    Batch 200 Loss 0.1009
    Batch 300 Loss 0.2371
    Batch 400 Loss 0.1779
    Batch 500 Loss 0.1999
    Train Loss 0.1634
    Validation Loss 0.5272
    Time taken for 1 epoch 40.04904556274414 sec
    
    Train Best Score : 0.16337499022483826
    ===============6============
    Batch 0 Loss 0.1121
    Batch 100 Loss 0.1140
    Batch 200 Loss 0.0643
    Batch 300 Loss 0.1265
    Batch 400 Loss 0.1043
    Batch 500 Loss 0.1580
    Train Loss 0.1139
    Validation Loss 0.5028
    Time taken for 1 epoch 37.852039098739624 sec
    
    Train Best Score : 0.11390714347362518
    ===============7============
    Batch 0 Loss 0.0448
    Batch 100 Loss 0.0681
    Batch 200 Loss 0.0861
    Batch 300 Loss 0.1097
    Batch 400 Loss 0.0746
    Batch 500 Loss 0.0589
    Train Loss 0.0885
    Validation Loss 0.5133
    Time taken for 1 epoch 38.54213619232178 sec
    
    Train Best Score : 0.0885130912065506
    ===============8============
    Batch 0 Loss 0.0591
    Batch 100 Loss 0.0973
    Batch 200 Loss 0.0531
    Batch 300 Loss 0.0440
    Batch 400 Loss 0.0683
    Batch 500 Loss 0.0557
    Train Loss 0.0731
    Validation Loss 0.5383
    Time taken for 1 epoch 37.85172772407532 sec
    
    Train Best Score : 0.0731188952922821
    ===============9============
    Batch 0 Loss 0.0450
    Batch 100 Loss 0.0563
    Batch 200 Loss 0.0194
    Batch 300 Loss 0.0517
    Batch 400 Loss 0.0223
    Batch 500 Loss 0.0656
    Train Loss 0.0681
    Validation Loss 0.5552
    Time taken for 1 epoch 37.873453855514526 sec
    
    Train Best Score : 0.06813544034957886
    ===============10============
    Batch 0 Loss 0.0578
    Batch 100 Loss 0.0279
    Batch 200 Loss 0.0716
    Batch 300 Loss 0.0573
    Batch 400 Loss 0.0991
    Batch 500 Loss 0.0820
    Train Loss 0.0670
    Validation Loss 0.5807
    Time taken for 1 epoch 37.90484857559204 sec
    
    Train Best Score : 0.06702984869480133
    ===============11============
    Batch 0 Loss 0.0335
    Batch 100 Loss 0.0280
    Batch 200 Loss 0.0767
    Batch 300 Loss 0.0756
    Batch 400 Loss 0.0746
    Batch 500 Loss 0.1033
    Train Loss 0.0684
    Validation Loss 0.5677
    Time taken for 1 epoch 37.88880634307861 sec
    
    Train Best Score : 0.06702984869480133
    ===============12============
    Batch 0 Loss 0.1446
    Batch 100 Loss 0.0575
    Batch 200 Loss 0.0952
    Batch 300 Loss 0.0598
    Batch 400 Loss 0.1535
    Batch 500 Loss 0.1103
    Train Loss 0.0659
    Validation Loss 0.5832
    Time taken for 1 epoch 37.880924701690674 sec
    
    Train Best Score : 0.0659254714846611
    ===============13============
    Batch 0 Loss 0.0556
    Batch 100 Loss 0.0524
    Batch 200 Loss 0.0525
    Batch 300 Loss 0.0838
    Batch 400 Loss 0.0408
    Batch 500 Loss 0.0603
    Train Loss 0.0645
    Validation Loss 0.5781
    Time taken for 1 epoch 37.7617506980896 sec
    
    Train Best Score : 0.06454772502183914
    ===============14============
    Batch 0 Loss 0.1319
    Batch 100 Loss 0.0319
    Batch 200 Loss 0.0504
    Batch 300 Loss 0.0367
    Batch 400 Loss 0.1865
    Batch 500 Loss 0.0554
    Train Loss 0.0637
    Validation Loss 0.5932
    Time taken for 1 epoch 37.82487726211548 sec
    
    Train Best Score : 0.06370946764945984
    ===============15============
    Batch 0 Loss 0.0277
    Batch 100 Loss 0.0813
    Batch 200 Loss 0.0524
    Batch 300 Loss 0.1030
    Batch 400 Loss 0.0920
    Batch 500 Loss 0.0459
    Train Loss 0.0638
    Validation Loss 0.6154
    Time taken for 1 epoch 37.93483757972717 sec
    
    Train Best Score : 0.06370946764945984
    ===============16============
    Batch 0 Loss 0.0604
    Batch 100 Loss 0.0694
    Batch 200 Loss 0.0784
    Batch 300 Loss 0.0495
    Batch 400 Loss 0.0286
    Batch 500 Loss 0.0784
    Train Loss 0.0634
    Validation Loss 0.6155
    Time taken for 1 epoch 37.84687852859497 sec
    
    Train Best Score : 0.06341858208179474
    ===============17============
    Batch 0 Loss 0.0363
    Batch 100 Loss 0.0870
    Batch 200 Loss 0.0721
    Batch 300 Loss 0.0527
    Batch 400 Loss 0.1034
    Batch 500 Loss 0.0412
    Train Loss 0.0609
    Validation Loss 0.6501
    Time taken for 1 epoch 37.78286647796631 sec
    
    Train Best Score : 0.060863640159368515
    ===============18============
    Batch 0 Loss 0.0248
    Batch 100 Loss 0.1002
    Batch 200 Loss 0.0564
    Batch 300 Loss 0.0379
    Batch 400 Loss 0.0552
    Batch 500 Loss 0.0983
    Train Loss 0.0571
    Validation Loss 0.6322
    Time taken for 1 epoch 37.90365505218506 sec
    
    Train Best Score : 0.05707473307847977
    ===============19============
    Batch 0 Loss 0.0295
    Batch 100 Loss 0.0253
    Batch 200 Loss 0.0320
    Batch 300 Loss 0.0959
    Batch 400 Loss 0.0304
    Batch 500 Loss 0.1387
    Train Loss 0.0566
    Validation Loss 0.6617
    Time taken for 1 epoch 37.80851078033447 sec
    
    Train Best Score : 0.05662443861365318
    ===============20============
    Batch 0 Loss 0.0858
    Batch 100 Loss 0.0198
    Batch 200 Loss 0.0502
    Batch 300 Loss 0.0251
    Batch 400 Loss 0.1950
    Batch 500 Loss 0.1086
    Train Loss 0.0592
    Validation Loss 0.6446
    Time taken for 1 epoch 37.817365407943726 sec
    
    Train Best Score : 0.05662443861365318
    ===============21============
    Batch 0 Loss 0.0854
    Batch 100 Loss 0.0235
    Batch 200 Loss 0.0760
    Batch 300 Loss 0.0187
    Batch 400 Loss 0.0508
    Batch 500 Loss 0.0833
    Train Loss 0.0623
    Validation Loss 0.6642
    Time taken for 1 epoch 37.84802961349487 sec
    
    Train Best Score : 0.05662443861365318
    ===============22============
    Batch 0 Loss 0.0269
    Batch 100 Loss 0.0525
    Batch 200 Loss 0.0278
    Batch 300 Loss 0.0711
    Batch 400 Loss 0.1088
    Batch 500 Loss 0.0547
    Train Loss 0.0598
    Validation Loss 0.6550
    Time taken for 1 epoch 37.9556143283844 sec
    
    Train Best Score : 0.05662443861365318
    ===============23============
    Batch 0 Loss 0.0269
    Batch 100 Loss 0.0263
    Batch 200 Loss 0.0433
    Batch 300 Loss 0.0422
    Batch 400 Loss 0.0258
    Batch 500 Loss 0.0922
    Train Loss 0.0586
    Validation Loss 0.6771
    Time taken for 1 epoch 37.98032832145691 sec
    
    Train Best Score : 0.05662443861365318
    ===============24============
    Batch 0 Loss 0.0249
    Batch 100 Loss 0.1731
    Batch 200 Loss 0.0245
    Batch 300 Loss 0.1095
    Batch 400 Loss 0.0948
    Batch 500 Loss 0.0219
    Train Loss 0.0582
    Validation Loss 0.6675
    Time taken for 1 epoch 37.85367560386658 sec
    
    Train Best Score : 0.05662443861365318
    ===============25============
    Batch 0 Loss 0.0268
    Batch 100 Loss 0.0370
    Batch 200 Loss 0.0531
    Batch 300 Loss 0.0245
    Batch 400 Loss 0.1633
    Batch 500 Loss 0.0393
    Train Loss 0.0609
    Validation Loss 0.6687
    Time taken for 1 epoch 37.798808336257935 sec
    
    Train Best Score : 0.05662443861365318
    ===============26============
    Batch 0 Loss 0.1464
    Batch 100 Loss 0.0271
    Batch 200 Loss 0.0572
    Batch 300 Loss 0.0306
    Batch 400 Loss 0.0726
    Batch 500 Loss 0.0853
    Train Loss 0.0613
    Validation Loss 0.6646
    Time taken for 1 epoch 37.848493576049805 sec
    
    Train Best Score : 0.05662443861365318
    ===============27============
    Batch 0 Loss 0.0482
    Batch 100 Loss 0.0713
    Batch 200 Loss 0.0783
    Batch 300 Loss 0.1146
    Batch 400 Loss 0.0430
    Batch 500 Loss 0.1291
    Train Loss 0.0599
    Validation Loss 0.6964
    Time taken for 1 epoch 37.85866045951843 sec
    
    Train Best Score : 0.05662443861365318
    ===============28============
    Batch 0 Loss 0.0347
    Batch 100 Loss 0.0229
    Batch 200 Loss 0.0521
    Batch 300 Loss 0.0421
    Batch 400 Loss 0.1166
    Batch 500 Loss 0.0254
    Train Loss 0.0579
    Validation Loss 0.6712
    Time taken for 1 epoch 37.9217894077301 sec
    
    Train Best Score : 0.05662443861365318
    ===============29============
    Batch 0 Loss 0.0375
    Batch 100 Loss 0.0215
    Batch 200 Loss 0.0341
    Batch 300 Loss 0.0309
    Batch 400 Loss 0.0473
    Batch 500 Loss 0.0193
    Train Loss 0.0573
    Validation Loss 0.6800
    Time taken for 1 epoch 37.970842123031616 sec
    
    Train Best Score : 0.05662443861365318
    ===============30============
    Batch 0 Loss 0.0134
    Batch 100 Loss 0.0295
    Batch 200 Loss 0.0289
    Batch 300 Loss 0.0476
    Batch 400 Loss 0.0463
    Batch 500 Loss 0.1034
    Train Loss 0.0612
    Validation Loss 0.7045
    Time taken for 1 epoch 37.9483916759491 sec
    
    Train Best Score : 0.05662443861365318
    


```python
x = [t for t in range(1, epochs+1)]

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

# 다중 플롯을 지정 : ax1은 y1에 대한 그래프
fig, ax1 = plt.subplots()
plt.title('Basic NMT')
ax1.plot(x, train_loss_list, color = 'red', alpha = 0.5)
ax1.set_ylabel('train score', color = 'red', rotation = 90)

# ax2는 y2에 대한 그래프, twinx로 x축을 공유
ax2 = ax1.twinx()
ax2.plot(x, valid_loss_list, color = 'blue', alpha = 0.5)
ax2.set_ylabel('valid score', color = 'blue', rotation = 90)

plt.show()
```


    
![png](/images/2023-02-06-Effecitve_Approaches_to_Attention-based_Neural_Machine_Translation_files/2023-02-06-Effecitve_Approaches_to_Attention-based_Neural_Machine_Translation_45_0.png)
    


### 3.2 NMT with Attention


```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + hidden_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # used for attention
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, enc_output):
        enc_output = enc_output.permute(1,0,2)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = torch.softmax(self.V(score), dim=1)

        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        output = self.embedding(input)

        # 64, 1, 1280
        output = torch.cat((context_vector.unsqueeze(1), output), -1)
        output, state = self.gru(output)
        output = self.fc(output.view(-1, output.size(2)))
        return output, state, attention_weights

    def initHidden(self, bt_size):
        return torch.zeros(1, bt_size, self.hidden_size, device=device)
```


```python
hidden_size = 1024
embedding_dim = 256

encoder = EncoderRNN(input_lang.n_words, embedding_dim, hidden_size).to(device)
att_decoder = AttnDecoderRNN(output_lang.n_words,embedding_dim, hidden_size).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(att_decoder.parameters()), 
                       lr=0.001)

criterion = nn.CrossEntropyLoss()

```


```python
teacher_forcing_ratio = 0.5

epochs = 30
best_score = 9999
train_loss_list = []
valid_loss_list = []
for epoch in range(epochs):
    print('==============={}============'.format(epoch+1))

    start = time.time()
    train_loss = train(encoder, att_decoder, optimizer, data_loader = train_loader)
    valid_loss = validation(encoder, att_decoder, optimizer,  data_loader = vali_loader)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    if best_score > train_loss:
        best_score = train_loss
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Train Best Score :', best_score)
```

    ===============1============
    Batch 0 Loss 3.8308
    Batch 100 Loss 1.6818
    Batch 200 Loss 1.1156
    Batch 300 Loss 1.2436
    Batch 400 Loss 0.9411
    Batch 500 Loss 1.1230
    Train Loss 1.2103
    Validation Loss 0.8272
    Time taken for 1 epoch 80.38685393333435 sec
    
    Train Best Score : 1.210310935974121
    ===============2============
    Batch 0 Loss 0.7302
    Batch 100 Loss 0.5611
    Batch 200 Loss 0.6322
    Batch 300 Loss 0.5187
    Batch 400 Loss 0.4823
    Batch 500 Loss 0.5093
    Train Loss 0.6281
    Validation Loss 0.5853
    Time taken for 1 epoch 80.00663590431213 sec
    
    Train Best Score : 0.6281476616859436
    ===============3============
    Batch 0 Loss 0.5256
    Batch 100 Loss 0.3975
    Batch 200 Loss 0.5117
    Batch 300 Loss 0.3379
    Batch 400 Loss 0.5227
    Batch 500 Loss 0.3000
    Train Loss 0.4006
    Validation Loss 0.5202
    Time taken for 1 epoch 79.86263918876648 sec
    
    Train Best Score : 0.40062639117240906
    ===============4============
    Batch 0 Loss 0.1721
    Batch 100 Loss 0.3314
    Batch 200 Loss 0.3671
    Batch 300 Loss 0.1909
    Batch 400 Loss 0.3911
    Batch 500 Loss 0.1714
    Train Loss 0.2746
    Validation Loss 0.5024
    Time taken for 1 epoch 79.94381594657898 sec
    
    Train Best Score : 0.2745780944824219
    ===============5============
    Batch 0 Loss 0.1663
    Batch 100 Loss 0.2691
    Batch 200 Loss 0.2034
    Batch 300 Loss 0.2155
    Batch 400 Loss 0.1644
    Batch 500 Loss 0.3874
    Train Loss 0.1957
    Validation Loss 0.4784
    Time taken for 1 epoch 79.91109466552734 sec
    
    Train Best Score : 0.19572710990905762
    ===============6============
    Batch 0 Loss 0.0971
    Batch 100 Loss 0.1645
    Batch 200 Loss 0.1714
    Batch 300 Loss 0.1342
    Batch 400 Loss 0.1790
    Batch 500 Loss 0.0692
    Train Loss 0.1515
    Validation Loss 0.5043
    Time taken for 1 epoch 79.89359927177429 sec
    
    Train Best Score : 0.15145964920520782
    ===============7============
    Batch 0 Loss 0.1588
    Batch 100 Loss 0.1144
    Batch 200 Loss 0.0614
    Batch 300 Loss 0.0833
    Batch 400 Loss 0.1766
    Batch 500 Loss 0.1761
    Train Loss 0.1222
    Validation Loss 0.4914
    Time taken for 1 epoch 79.88205599784851 sec
    
    Train Best Score : 0.12222021073102951
    ===============8============
    Batch 0 Loss 0.0941
    Batch 100 Loss 0.1591
    Batch 200 Loss 0.1743
    Batch 300 Loss 0.1482
    Batch 400 Loss 0.1606
    Batch 500 Loss 0.1986
    Train Loss 0.1048
    Validation Loss 0.5232
    Time taken for 1 epoch 80.0357894897461 sec
    
    Train Best Score : 0.104762502014637
    ===============9============
    Batch 0 Loss 0.1822
    Batch 100 Loss 0.0569
    Batch 200 Loss 0.1835
    Batch 300 Loss 0.0577
    Batch 400 Loss 0.0819
    Batch 500 Loss 0.1390
    Train Loss 0.0912
    Validation Loss 0.4984
    Time taken for 1 epoch 79.83175468444824 sec
    
    Train Best Score : 0.09117243438959122
    ===============10============
    Batch 0 Loss 0.1173
    Batch 100 Loss 0.0308
    Batch 200 Loss 0.0961
    Batch 300 Loss 0.1027
    Batch 400 Loss 0.0905
    Batch 500 Loss 0.1270
    Train Loss 0.0837
    Validation Loss 0.5365
    Time taken for 1 epoch 79.9092948436737 sec
    
    Train Best Score : 0.08372192829847336
    ===============11============
    Batch 0 Loss 0.0527
    Batch 100 Loss 0.0887
    Batch 200 Loss 0.0684
    Batch 300 Loss 0.1392
    Batch 400 Loss 0.0806
    Batch 500 Loss 0.0972
    Train Loss 0.0801
    Validation Loss 0.5161
    Time taken for 1 epoch 79.69076633453369 sec
    
    Train Best Score : 0.08013681322336197
    ===============12============
    Batch 0 Loss 0.1380
    Batch 100 Loss 0.0684
    Batch 200 Loss 0.0609
    Batch 300 Loss 0.0664
    Batch 400 Loss 0.0417
    Batch 500 Loss 0.0823
    Train Loss 0.0722
    Validation Loss 0.5338
    Time taken for 1 epoch 79.75847578048706 sec
    
    Train Best Score : 0.07222533971071243
    ===============13============
    Batch 0 Loss 0.0418
    Batch 100 Loss 0.0215
    Batch 200 Loss 0.0479
    Batch 300 Loss 0.0644
    Batch 400 Loss 0.0979
    Batch 500 Loss 0.0458
    Train Loss 0.0743
    Validation Loss 0.5463
    Time taken for 1 epoch 79.88189101219177 sec
    
    Train Best Score : 0.07222533971071243
    ===============14============
    Batch 0 Loss 0.0664
    Batch 100 Loss 0.0410
    Batch 200 Loss 0.0553
    Batch 300 Loss 0.0423
    Batch 400 Loss 0.0915
    Batch 500 Loss 0.0445
    Train Loss 0.0712
    Validation Loss 0.5465
    Time taken for 1 epoch 79.6895067691803 sec
    
    Train Best Score : 0.0711677223443985
    ===============15============
    Batch 0 Loss 0.0939
    Batch 100 Loss 0.0236
    Batch 200 Loss 0.0745
    Batch 300 Loss 0.1229
    Batch 400 Loss 0.0293
    Batch 500 Loss 0.0559
    Train Loss 0.0685
    Validation Loss 0.5680
    Time taken for 1 epoch 79.64896249771118 sec
    
    Train Best Score : 0.06850026547908783
    ===============16============
    Batch 0 Loss 0.0379
    Batch 100 Loss 0.0468
    Batch 200 Loss 0.0238
    Batch 300 Loss 0.0788
    Batch 400 Loss 0.0363
    Batch 500 Loss 0.0387
    Train Loss 0.0631
    Validation Loss 0.5659
    Time taken for 1 epoch 79.75520277023315 sec
    
    Train Best Score : 0.06306355446577072
    ===============17============
    Batch 0 Loss 0.0344
    Batch 100 Loss 0.0344
    Batch 200 Loss 0.0911
    Batch 300 Loss 0.0454
    Batch 400 Loss 0.1203
    Batch 500 Loss 0.0263
    Train Loss 0.0629
    Validation Loss 0.5757
    Time taken for 1 epoch 79.82434964179993 sec
    
    Train Best Score : 0.06291705369949341
    ===============18============
    Batch 0 Loss 0.0342
    Batch 100 Loss 0.0308
    Batch 200 Loss 0.0439
    Batch 300 Loss 0.0636
    Batch 400 Loss 0.0595
    Batch 500 Loss 0.1044
    Train Loss 0.0619
    Validation Loss 0.5792
    Time taken for 1 epoch 79.83721923828125 sec
    
    Train Best Score : 0.06185527145862579
    ===============19============
    Batch 0 Loss 0.0747
    Batch 100 Loss 0.0506
    Batch 200 Loss 0.0947
    Batch 300 Loss 0.1002
    Batch 400 Loss 0.0570
    Batch 500 Loss 0.0300
    Train Loss 0.0614
    Validation Loss 0.5821
    Time taken for 1 epoch 79.71416735649109 sec
    
    Train Best Score : 0.06142011284828186
    ===============20============
    Batch 0 Loss 0.0275
    Batch 100 Loss 0.0860
    Batch 200 Loss 0.1002
    Batch 300 Loss 0.1435
    Batch 400 Loss 0.0585
    Batch 500 Loss 0.0747
    Train Loss 0.0571
    Validation Loss 0.5909
    Time taken for 1 epoch 79.46572184562683 sec
    
    Train Best Score : 0.05710778385400772
    ===============21============
    Batch 0 Loss 0.0600
    Batch 100 Loss 0.0235
    Batch 200 Loss 0.0316
    Batch 300 Loss 0.0558
    Batch 400 Loss 0.1049
    Batch 500 Loss 0.0478
    Train Loss 0.0577
    Validation Loss 0.5653
    Time taken for 1 epoch 79.73875117301941 sec
    
    Train Best Score : 0.05710778385400772
    ===============22============
    Batch 0 Loss 0.0307
    Batch 100 Loss 0.0903
    Batch 200 Loss 0.0173
    Batch 300 Loss 0.0591
    Batch 400 Loss 0.0384
    Batch 500 Loss 0.0300
    Train Loss 0.0539
    Validation Loss 0.5971
    Time taken for 1 epoch 79.79578948020935 sec
    
    Train Best Score : 0.05386403948068619
    ===============23============
    Batch 0 Loss 0.0168
    Batch 100 Loss 0.0356
    Batch 200 Loss 0.0304
    Batch 300 Loss 0.0277
    Batch 400 Loss 0.0509
    Batch 500 Loss 0.0386
    Train Loss 0.0524
    Validation Loss 0.5963
    Time taken for 1 epoch 79.55818271636963 sec
    
    Train Best Score : 0.05242343246936798
    ===============24============
    Batch 0 Loss 0.0785
    Batch 100 Loss 0.0167
    Batch 200 Loss 0.0222
    Batch 300 Loss 0.0726
    Batch 400 Loss 0.0440
    Batch 500 Loss 0.1988
    Train Loss 0.0550
    Validation Loss 0.5930
    Time taken for 1 epoch 79.68427681922913 sec
    
    Train Best Score : 0.05242343246936798
    ===============25============
    Batch 0 Loss 0.0313
    Batch 100 Loss 0.0626
    Batch 200 Loss 0.0583
    Batch 300 Loss 0.0193
    Batch 400 Loss 0.0456
    Batch 500 Loss 0.0371
    Train Loss 0.0540
    Validation Loss 0.5933
    Time taken for 1 epoch 79.6644675731659 sec
    
    Train Best Score : 0.05242343246936798
    ===============26============
    Batch 0 Loss 0.0576
    Batch 100 Loss 0.0354
    Batch 200 Loss 0.0723
    Batch 300 Loss 0.0526
    Batch 400 Loss 0.0261
    Batch 500 Loss 0.0895
    Train Loss 0.0540
    Validation Loss 0.6010
    Time taken for 1 epoch 79.47383141517639 sec
    
    Train Best Score : 0.05242343246936798
    ===============27============
    Batch 0 Loss 0.0755
    Batch 100 Loss 0.0403
    Batch 200 Loss 0.0432
    Batch 300 Loss 0.0260
    Batch 400 Loss 0.0687
    Batch 500 Loss 0.0458
    Train Loss 0.0561
    Validation Loss 0.5943
    Time taken for 1 epoch 79.65788006782532 sec
    
    Train Best Score : 0.05242343246936798
    ===============28============
    Batch 0 Loss 0.0647
    Batch 100 Loss 0.0158
    Batch 200 Loss 0.0375
    Batch 300 Loss 0.0710
    Batch 400 Loss 0.0609
    Batch 500 Loss 0.0951
    Train Loss 0.0512
    Validation Loss 0.6249
    Time taken for 1 epoch 79.6898844242096 sec
    
    Train Best Score : 0.05118886008858681
    ===============29============
    Batch 0 Loss 0.0137
    Batch 100 Loss 0.0388
    Batch 200 Loss 0.0219
    Batch 300 Loss 0.0142
    Batch 400 Loss 0.0544
    Batch 500 Loss 0.0853
    Train Loss 0.0513
    Validation Loss 0.6104
    Time taken for 1 epoch 79.6860122680664 sec
    
    Train Best Score : 0.05118886008858681
    ===============30============
    Batch 0 Loss 0.0244
    Batch 100 Loss 0.0326
    Batch 200 Loss 0.0407
    Batch 300 Loss 0.1039
    Batch 400 Loss 0.0217
    Batch 500 Loss 0.0307
    Train Loss 0.0496
    Validation Loss 0.6027
    Time taken for 1 epoch 79.69880867004395 sec
    
    Train Best Score : 0.049604784697294235
    


```python
x = [t for t in range(1, epochs+1)]

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

# 다중 플롯을 지정 : ax1은 y1에 대한 그래프
fig, ax1 = plt.subplots()
plt.title('NMT with Attention')
ax1.plot(x, train_loss_list, color = 'red', alpha = 0.5)
ax1.set_ylabel('train score', color = 'red', rotation = 90)

# ax2는 y2에 대한 그래프, twinx로 x축을 공유
ax2 = ax1.twinx()
ax2.plot(x, valid_loss_list, color = 'blue', alpha = 0.5)
ax2.set_ylabel('valid score', color = 'blue', rotation = 90)

plt.show()
```


    
![png](/images/2023-02-06-Effecitve_Approaches_to_Attention-based_Neural_Machine_Translation_files/2023-02-06-Effecitve_Approaches_to_Attention-based_Neural_Machine_Translation_50_0.png)
    


## 4. Evaluation


```python
def evaluate(encoder, decoder, idx, max_length=MAX_LENGTH):
    inp = input_tensor[idx].unsqueeze(0)
    lens = torch.tensor(list(map(lambda x: torch.sum(x!=0).item(),inp)))
    batch = 1

    lens, indx = lens.sort(dim=0, descending=True)
    inp = inp.transpose(0,1)
    max_length = 20


    with torch.no_grad():
        enc_output, enc_hidden = encoder(inp.to(device), lens, batch)
        dec_hidden = enc_hidden
        
        decoder_input = torch.tensor([[SOS_token]]*batch, device=device)
        decoder_attentions = torch.zeros(max_length, max_length)
        decoded_words = []

        for t in range(0, max_length):
            decoder_output, dec_hidden, decoder_attention = att_decoder(
                decoder_input, dec_hidden, enc_output)
            att_output = decoder_attention[0].squeeze(1)
            decoder_attentions[t][:att_output.shape[0]] = att_output
            topv, topi = decoder_output.topk(1)
            if topi.item() == 0:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi

        return decoded_words, decoder_attentions[:t + 1]

```


```python
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        idx = random.choice(range(input.shape[0]))
        print('>', input[idx])
        print('=', target[idx])
        output_words, attentions = evaluate(encoder, att_decoder, idx, MAX_LENGTH)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
```


```python
evaluateRandomly(encoder, att_decoder)
```

    > j ai peur de ce que l instituteur dira .
    = i am afraid of what the teacher will say .
    < i am afraid of what the teacher will say . EOS
    
    > si vous n en voulez pas je la donnerai a quelqu un d autre .
    = if you don t want this i ll give it to someone else .
    < if you don t want it i ll give it to someone else . EOS
    
    > pensez vous serieusement a divorcer ?
    = are you seriously thinking about getting a divorce ?
    < are you seriously thinking about getting a lot of that ? EOS
    
    > personne ne va te faire de mal .
    = no one is going to harm you .
    < no one is going to harm you . EOS
    
    > tu vas adorer notre nourriture .
    = you re going to love our food .
    < you re going to love with our place . EOS
    
    > c etait un grand musicien .
    = he was a great musician .
    < he was a great musician . EOS
    
    > tom s inquiete pour ses enfants .
    = tom is worried about his children .
    < tom is anxious about her children . EOS
    
    > j ai du mal a me concentrer .
    = i m having trouble focusing .
    < i m having trouble focusing . EOS
    
    > nous avons fini de dejeuner .
    = we have finished lunch .
    < we have finished lunch . EOS
    
    > tu aurais du ainsi faire .
    = you should have done so .
    < you should have done so . EOS
    
    


```python
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder, att_decoder, idx)
    print('input =', input[idx])
    print('output =', ' '.join(output_words))
    showAttention(input[idx], output_words, attentions)

```


```python
idx = random.choice(range(input.shape[0]))
evaluateAndShowAttention(idx)
```

    input = je sais pourquoi vous l avez fait .
    output = i know why you did it . EOS
    


    
![png](/images/2023-02-06-Effecitve_Approaches_to_Attention-based_Neural_Machine_Translation_files/2023-02-06-Effecitve_Approaches_to_Attention-based_Neural_Machine_Translation_56_1.png)
    


# Reference
[1] https://blog.paperspace.com/seq2seq-translator-pytorch/

[2] https://github.com/omarsar/pytorch_neural_machine_translation_attention/blob/master/NMT_in_PyTorch.ipynb

[3] https://medium.com/syncedreview/a-brief-overview-of-attention-mechanism-13c578ba9129

[4] https://velog.io/@sjinu/%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%AC-Attention-Mechanism

[5] https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
