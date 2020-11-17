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
        dataset = []
        for line in open(self.data_pose,'r'):
            spl = line.strip().split('\t')
            dataset.append([spl[0]])
        dataset_np = np.array(dataset)
        
        self.img_name = dataset_np[:,0]
         
    def __len__(self): 
        return len(self.img_name)

    def __getitem__(self, idx):
        
        file_path = self.dir + self.img_name[idx]
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image

