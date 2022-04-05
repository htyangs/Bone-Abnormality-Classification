import os
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve,auc
import re
import datetime
from torch.optim import lr_scheduler
import argparse
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
parser.add_argument("-output ", "--outpath")
args = parser.parse_args()

train_csv = os.path.join(args.datapath, 'train.csv')
train_dir = os.path.join(args.datapath,  'train')
test_dir = os.path.join(args.datapath,  'test')
test_csv = args.outpath
seed = 42
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(seed)

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform, train=True, csv=None):
        self.train = train
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
        if train and csv is None:
          raise TypeError('csv is none.')
        self.label = csv.set_index('id') if train else None

    def __getitem__(self, idx):
        self.photo_num = 0
        fname = self.fnames[idx]
        if("HAND" in fname):
            id_name = fname[-26:-11]
        elif("WRIST" in fname):
            id_name = fname[-27:-11]
        elif("FOREARM" in fname):
            id_name = fname[-29:-11]
        num = fname[-5]
        img = self.check_image_num(fname,num,True)
        if self.train:
            return img, self.label.loc[id_name][0],fname[-29:-4]
        else:
            return img, fname 
    

        
    #transform the image
    def check_image_num (self,fname,image_num,first_time): 
        img2_path = fname
        if os.path.exists(img2_path):
            img2 = torchvision.io.read_image(img2_path,torchvision.io.ImageReadMode(1))
            img2 = self.transform(img2)
            padding2 = (int((512-img2.size()[2])/2),512-img2.size()[2]- int((512-img2.size()[2])/2), int((512-img2.size()[1])/2),512-img2.size()[1]-int((512-img2.size()[1])/2))
            img2 = F.pad(input=img2, pad=padding2, mode='constant', value=0)
            return img2
        else:
            print('error')
        

    def __len__(self):
        return self.num_samples


def get_dataset(root, train, csv=None):
    fnames = glob.glob(os.path.join(root, '*') )
    #check if is nan
    if train:
        if csv is None:
          raise TypeError('csv is none.')
        
        #chech if the image is nan
        label = csv
        labed_data = label['id'][label['label']==label['label']]
        label_name = label['label'][label['label']==label['label']]
        labed_data = labed_data.values
        fnames_2 = fnames.copy() 

        #remove nan label data and forearm     
        for i in range(len(fnames_2)):
          if(fnames_2[i][-26:-11] not in labed_data): # hand
            if(fnames_2[i][-27:-11] not in labed_data): #add wrist
              fnames.remove(fnames_2[i])
              
    train_compose = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-30,30), translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485, std=0.229)
    ]
    test_compose = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485, std=0.229)
    ]
    transform = transforms.Compose(train_compose if train else test_compose)
    dataset = CrypkoDataset(fnames, transform, train, csv)
    return dataset

csv = pd.read_csv(train_csv)
test_dataset = get_dataset(test_dir, train=False, csv=csv.copy())
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


name = 'regnet'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_name = os.path.join(args.datapath, str(6)+name+'.pt')
model = torch.load(save_name)
csv_name = test_csv
anomality = {}
label = torch.tensor([1.0]).type(torch.ByteTensor).to(device)
count = 2000
i=0
with torch.no_grad():
    for inputs, fname in test_loader:
        #inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device).unsqueeze(1)
        i=i+1
        inputs = inputs.to(device)
        output = model.forward(inputs)
        loss = output.squeeze()
        iname = os.path.basename(fname[0])[:-11]
        print(i,"/1384:",iname,loss.item())
        if iname not in anomality:
            anomality[iname] = []
        else:
            print('same',iname)
        anomality[iname].append(loss.cpu().data.numpy())
        #print(loss[0].item)
        
        count-=1
        if count == 0 :
             break


import csv
with open(csv_name, 'w', newline='') as result:
    writer = csv.writer(result)
    writer.writerow(['id', 'label'])
    for iname, lossarray in anomality.items():
        writer.writerow([iname, np.mean(lossarray)])