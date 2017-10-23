
import torch 
import torch.nn as nn 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import os 
from torch.autograd import Variable 


class data_loader(Dataset):
    def __init__(self, path, transform=False, file_suffix = "*.jpg", label = None, test = False):
        super().__init__()
        self.path = path
        self.files = glob.glob(os.path.join(path ,"*.jpg"))
        self.transform = transform
        # self.label = torch.cuda.LongTensor(label)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = cv2.imread(img_name)
        img = cv2.resize(img, dsize = (200,200))
        img = np.einsum('ijk->kij', img)
        img = torch.from_numpy(img)
        if test:
            print(img_name)
#         img = torch.FloatTensor(img)
        # d = {'image'  : img, 'name' : img_name, 'label' : self.label }
        label = os.path.basename(img_name).split('.')[0] == 'cat' if 0 else  1
        return img, label



# will do it later 
#  transform = pass


