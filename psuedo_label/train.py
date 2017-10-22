from dataset.dataset import data_loader
from model import cnn_basic
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn 
import numpy as np 

data_path_train = None 
data_path_test = None
data_path_valid = None

train_loader = data_loader(path = data_path_train)
valid_loader = data_loader(path = data_path_valid)
test_loader = data_loader(path = data_path_test)


train_loader = DataLoader(train_loader,
                        batch_size = 32,
                        shuffle = True,
                        num_workers = 4
                        #pin_memory = True  # Uncomment if only using CUDA  
                        )

valid_loader = DataLoader(valid_loader,
                        batch_size = 32,
                        shuffle = True,
                        num_workers = 4
                        #pin_memory = True  # Uncomment if only using CUDA  
                        )

test_loader = DataLoader(test_loader,
                        batch_size = 32,
                        shuffle = True,
                        num_workers = 4
                        #pin_memory = True  # Uncomment if only using CUDA  
                        )


cnn = cnn_basic.CNN()
cnn.cuda()  # To make it run on GPU 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001)

loss = []

for epoch in range(num_epoch):
    for k, d in enumerate(loader):
        img = d['image']
        label = d['label']
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        # optimize initialize zero grad 
        optimizer.zero_grad()
        output = cnn(img.float())
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()