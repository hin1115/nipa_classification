import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms, utils
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
import PIL
from PIL import Image
import PIL.Image as pilimg



class Make_Dataset(Dataset): 
    def __init__(self, data_dir, transform=None):
        self.dir = data_dir
        self.transform = transform
        self.data_pose = data_dir + 'train.tsv'
        
        self.class_num_1 = 14
        self.class_num_2 = 21
        
        dataset = []
        for line in open(self.data_pose,'r'):
            spl = line.strip().split('\t')
            #dataset.append([spl[0]])
            dataset.append([spl[0], spl[1], spl[2]])
        dataset_np = np.array(dataset)
        self.img_name = dataset_np[:,0]
        self.cls_1 = dataset_np[:,1].astype(np.int)
        self.cls_2 = dataset_np[:,2].astype(np.int)
         
    def __len__(self): 
        return len(self.img_name)

    def __getitem__(self, idx):
        file_path = self.dir + self.img_name[idx]
        image = Image.open(file_path)
        img_arr = np.array(image) / 255.
        print("flag-2")
        if self.transform:
            image_transform = self.transform(img_arr)
            #image_transform = self.transform(image_arr)
        image.close()
        print("flag-3")
        
        
        x_input = torch.FloatTensor( np.transpose(image_transform, (2, 0, 1)) )
        
        # label - 1 part
        y1_lab_onehot = np.zeros(shape=(self.class_num_1,), dtype=np.int8)
        y1_lab_onehot[self.cls_1[idx]] = 1
        
        # label - 2 part
        y2_lab_onehot = np.zeros(shape=(self.class_num_2,), dtype=np.int8)
        y2_lab_onehot[self.cls_2[idx]] = 1
        
        x = torch.FloatTensor(x_input)
        y1 = torch.FloatTensor(y1_lab_onehot)
        y2 = torch.FloatTensor(y2_lab_onehot)
        return x, y1, y2
        
        
        
        return [image_transform, ]

