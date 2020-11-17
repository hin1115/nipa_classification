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


#import os 
#os.environ['CUDA_VISIBLE_DEVICES']='6,7'


class CustomDataset_train(Dataset): 
    def __init__(self, data_dir, class_num_1, class_num_2, valid_split, transforms=None):
        self.dir = data_dir
        self.data_pose = data_dir + 'train.tsv'
        # Need zero class >> plus one part
        self.class_num_1 = class_num_1 # number of first label
        self.class_num_2 = class_num_2 # number of second label
        self.enc = OneHotEncoder
        self.transforms = transforms
        self.valid_split = valid_split
        
        dataset = []
        for line in open(self.data_pose,'r'):
            spl = line.strip().split('\t')
            dataset.append([spl[0], spl[1], spl[2]])
        
        dataset_np = np.array(dataset)
        
        self.img_name = dataset_np[:,0]
        self.cls_1 = dataset_np[:,1].astype(np.int)
        self.cls_2 = dataset_np[:,2].astype(np.int)
        
    # 총 데이터의 개수를 리턴
    def __len__(self): 
        dataset_size = len(self.img_name)
        split = int(np.floor((1-self.valid_split) * dataset_size))
        return split

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        # image part
        img_name = self.dir + self.img_name[idx]
        img_data = pilimg.open(img_name)
        if self.transforms:
            x_input = self.transforms(img_data)
        #img_arr = np.array(x_input) / 255.
        # transform img_arr from [height, width, depth] to [depth, height, width]
        #x_input = np.transpose(img_arr, (2, 0, 1))
        x_input = np.array(x_input)
        img_data.close()
        
        # label - 1 part
        y1_lab_onehot = np.zeros(shape=(self.class_num_1,), dtype=np.int8)
        y1_lab_onehot[self.cls_1[idx]] = 1
        
        # label - 2 part
        y2_lab_onehot = np.zeros(shape=(self.class_num_2,), dtype=np.int8)
        y2_lab_onehot[self.cls_2[idx]] = 1
        
        #x = torch.FloatTensor(x_input)
        x = x_input
        y1 = torch.FloatTensor(y1_lab_onehot)
        y2 = torch.FloatTensor(y2_lab_onehot)
        
        return x, y1, y2

    
class CustomDataset_valid(Dataset): 
    def __init__(self, data_dir, class_num_1, class_num_2, valid_split, transforms=None):
        self.dir = data_dir
        self.data_pose = data_dir + 'train.tsv'
        # Need zero class >> plus one part
        self.class_num_1 = class_num_1 # number of first label
        self.class_num_2 = class_num_2 # number of second label
        self.enc = OneHotEncoder
        self.transforms = transforms
        self.valid_split = valid_split
        
        dataset = []
        for line in open(self.data_pose,'r'):
            spl = line.strip().split('\t')
            dataset.append([spl[0], spl[1], spl[2]])
        
        dataset_np = np.array(dataset)
        
        self.img_name = dataset_np[:,0]
        self.cls_1 = dataset_np[:,1].astype(np.int)
        self.cls_2 = dataset_np[:,2].astype(np.int)
        
    # 총 데이터의 개수를 리턴 : maybe there is some mistake >> overlap train and validation
    def __len__(self): 
        dataset_size = len(self.img_name)
        split = int(np.floor( self.valid_split * dataset_size))
        return split

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        # image part
        img_name = self.dir + self.img_name[idx]
        img_data = pilimg.open(img_name)
        if self.transforms:
            x_input = self.transforms(img_data)
        #img_arr = np.array(x_input) / 255.
        # transform img_arr from [height, width, depth] to [depth, height, width]
        #x_input = np.transpose(img_arr, (2, 0, 1))    
        x_input = np.array(x_input)
        img_data.close()
        
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