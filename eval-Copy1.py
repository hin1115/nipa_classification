from models.imagenet import resnext
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
from models.call_model import Net

#import csv
import pandas as pd
from pandas import DataFrame



os.environ['CUDA_VISIBLE_DEVICES']='5'

PATH = './result/'

if not os.path.isdir(PATH):
    os.mkdir(PATH)

# define the hyper parameters for training
nb_batch = 8
nb_epochs = 3
lr = 1e-4
#random_seed= 42
#shuffle_dataset = True
img_path = '/dataset/test/'
checkpoint_path = 'result/ckpt_folder_1026/epoch_099_ckpt.pth'

class CustomDataset(Dataset): 
    def __init__(self, data_dir='test/', class_num_1=13, class_num_2=20):
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

#####

class _res_block(nn.Module):
    def __init__(self, in_channels):
        super(_res_block, self).__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()
        
        self.conv_2 = _res_block(in_channels=64)        
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.relu_3 = nn.ReLU()
        
        self.conv_4 = _res_block(in_channels=128)
        
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.relu_5 = nn.ReLU()
        
        self.conv_6 = _res_block(in_channels=256)
        
        self.conv_7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_7 = nn.BatchNorm2d(512)
        self.relu_7 = nn.ReLU()
        
        self.conv_8 = _res_block(in_channels=512)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc_1 = nn.Linear(512, 14)
        self.fc_2 = nn.Linear(512, 21)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        # stage - 1
        out = self.bn_1(self.conv_1(x))
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.maxpool(out)
        
        # stage - 2
        out = self.bn_3(self.conv_3(out))
        out = self.relu_3(out)
        out = self.conv_4(out)
        out = self.maxpool(out)
        
        # stage - 3
        out = self.bn_5(self.conv_5(out))
        out = self.relu_5(out)
        out = self.conv_6(out)
        out = self.maxpool(out)
        
        # stage - 4
        out = self.bn_7(self.conv_7(out))
        out = self.relu_7(out)
        out = self.conv_8(out)
        
        # glocal average pooling 
        out = self.gap(out)
        
        # fully connected layer and soft-max
        out_flat = out.view(out.size(0), -1)
        out_1 = self.fc_1(out_flat)
        out_2 = self.fc_2(out_flat)
        
        return self.softmax(out_1), self.softmax(out_2)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

model.load_state_dict( torch.load(checkpoint_path,map_location=device) )
model.eval()


dataset_test = CustomDataset()

test_loader = DataLoader(dataset_test, batch_size=nb_batch)
print("start testing=========================")

#f = open("temp_test.tsv", 'w', encoding='utf-8', newline='\n')
#f = open("test.tsv", 'w', newline='\n')
#wr = csv.writer(f, delimiter='\t')

#model.eval()
with torch.no_grad(): 
    test_loss = 0
    correct_1 = 0
    correct_2 = 0
    
    idx_num = 0
    save_tsv_file_list = []
    
    for batch_idx, samples in enumerate(test_loader):
        #x_test, y_test1, y_test2 = samples
        x_test, x_test_name = samples
        data = x_test.to(device)
        output_1, output_2 = model(data)
        pred_1 = output_1.argmax(dim=1, keepdim=True)
        pred_2 = output_2.argmax(dim=1, keepdim=True)
        #print("validation accuracy : case - 1 : {:.6f} // case-2 : {:.6f}".format(corr_1_val, corr_2_val) )

        #print("\t iteration num : {} // index - 1 : {} // index - 1 : {}".format(batch_idx, pred_1, pred_2))
        
        pred_1_arr = pred_1.cpu().numpy()
        pred_2_arr = pred_2.cpu().numpy()
        for b_idx in range( len(x_test_name) ):
            a=pred_1_arr[b_idx][0].astype(str)
            b=pred_2_arr[b_idx][0].astype(str)
            save_tsv_file_list.append( [ x_test_name[b_idx], a,b])
            #save_tsv_file_list.append( [ x_test_name[b_idx], pred_1_arr[b_idx][0], pred_2_arr[b_idx][0] ])
            
            
            #save_tsv_file_list.append( [ pred_1_arr[b_idx][0].astype(int), pred_2_arr[b_idx][0].astype(int) ])
            #wr.writerow( [x_test_name[b_idx], str(pred_1_arr[b_idx][0]), str(pred_2_arr[b_idx][0])] )

        idx_num += nb_batch
        print("index check~!! ========================")
        print( "iter num : {} // entire : {}".format(idx_num,len(dataset_test)) )

        
        

#f.close()

data_df = DataFrame(save_tsv_file_list)

data_df.to_csv("test_49.tsv", index=False, header=None, sep="\t")
        
#np.savetxt( "temp_test.tsv", save_tsv_file_list, fmt="%s %s %s", delimiter='\t', newline='\n')        
#np.savetxt( "temp_test.tsv", save_tsv_file_list, fmt="%s %s %s", delimiter='\t')
        

