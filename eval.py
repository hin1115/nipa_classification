
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
from utils.label_eval import CustomDataset
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
class_num_1 = 13
class_num_2 = 20
#random_seed= 42
#shuffle_dataset = True
data_dir = './dataset/test/'

checkpoint_path = 'result/ckpt_folder_1027/epoch_046_ckpt.pth'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

model.load_state_dict( torch.load(checkpoint_path,map_location=device) )
model.eval()


dataset_test = CustomDataset(data_dir, class_num_1,class_num_2)

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


data_df.to_csv("./result/test_4.tsv", index=False, header=None, sep="\t")
        
#np.savetxt( "temp_test.tsv", save_tsv_file_list, fmt="%s %s %s", delimiter='\t', newline='\n')        
#np.savetxt( "temp_test.tsv", save_tsv_file_list, fmt="%s %s %s", delimiter='\t')
        

