#from trasnfrom import
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import sklearn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import PIL
from PIL import Image
import PIL.Image as pilimg
import torch.nn as nn

class CustomDataset(Dataset): 
    def __init__(self, data_dir, class_num_1, class_num_2):
        self.dir = data_dir
        self.data_pose = data_dir + 'test.tsv'
        # Need zero class >> plus one part
        self.class_num_1 = class_num_1 + 1 # number of first label
        self.class_num_2 = class_num_2 + 1 # number of second label
        self.enc = OneHotEncoder
        
        dataset = []    
                
        for line in open(self.data_pose,'r'):
            spl = line.strip().split('\t')
            dataset.append([spl[0]])            
            
        dataset_np = np.array(dataset)
        
        self.img_name = dataset_np[:,0]

        
    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.img_name)
        #return len(self.img_name[:32])

    
    def __getitem__(self, idx): 
        img_name = self.dir + self.img_name[idx]
        img_data = pilimg.open(img_name)
        img_arr = np.array(img_data) / 255.
        x_input = torch.FloatTensor( np.transpose(img_arr, (2, 0, 1)) )

        x = torch.FloatTensor(x_input)
        return x, self.img_name[idx]