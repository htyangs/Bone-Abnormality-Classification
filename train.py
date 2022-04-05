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


parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
args = parser.parse_args()

train_csv = os.path.join(args.datapath, 'train.csv')
train_dir = os.path.join(args.datapath,  'train')
test_dir = os.path.join(args.datapath,  'test')

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
train_dataset = get_dataset(train_dir, train=True, csv=csv.copy())
test_dataset = get_dataset(test_dir, train=False, csv=csv.copy())

VAL_RATIO = 0.2
batch_size = 32

#get weight of subset
def get_weight2(subset):
    
    weight = np.array([])
    count_0 = 0

    for name_id in subset.indices:
        name = subset.dataset.fnames[name_id]        
        fullfilename = os.path.basename(name)
        id_name = re.search('^(.*)_.*?$', fullfilename).group(1) 
        # HAND_7c61ce2000_image3.png -> 'HAND_7c61ce2000'
        
        my_label = subset.dataset.label.loc[id_name][0]

        if not my_label == 1:
            count_0 += 1

        weight = np.append(weight, my_label)

    weight[weight==1] = count_0 / len(subset)
    weight[weight==0] = 1.0 - count_0 / len(subset)

    return weight
    
# Build dataloader
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
percent = int(len(train_dataset) * (1 - VAL_RATIO))
train_set, valid_set = data.random_split(train_dataset, [percent, len(train_dataset)-percent])
weight = get_weight2(train_set)
sampler = WeightedRandomSampler(weight,len(train_set))
train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
val_weight = get_weight2(valid_set)
val_sampler = WeightedRandomSampler(val_weight,len(valid_set))
val_loader = DataLoader(dataset=valid_set, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
weight_loss = torch.cuda.FloatTensor([np.max(weight)/(1-np.max(weight))])

# Get pretrained model using torchvision.models as models library
model = models.regnet_x_800mf(pretrained=True)
model_weight = model.stem[0].weight.clone()
input_layer  = nn.Conv2d(1, model.stem[0].out_channels, kernel_size=model.stem[0].kernel_size, stride=model.stem[0].stride, padding=(3, 3), bias=False).requires_grad_()
input_layer.weight[:,:1,:,:].data[...] =  Variable(model_weight[:,:1,:,:], requires_grad=True)
model.stem[0] = input_layer

for param in model.parameters():
    param.requires_grad = True

epochs = 20
num_labels = 1 
classifier = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                           nn.BatchNorm1d(1024),
                           nn.Dropout(0.3),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.BatchNorm1d(512),
                           nn.Dropout(0.3),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.Sigmoid())

model.fc = classifier
BCELoss = nn.BCELoss(weight_loss)
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(),lr=10**-4)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

min_val_loss = 100
min_epoch = 0
name = 'regnet'
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels,names in train_loader:
        inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model.forward(inputs)
        trainloss = BCELoss(output, labels)
        train_loss += trainloss.item()*inputs.size(0)
        trainloss.backward()
        optimizer.step()

    labelarr = torch.tensor([]).to(device)
    outputarr =  torch.tensor([]).to(device)
    model.eval()
    counter = 0
    # Validation
    with torch.no_grad():
        for inputs, labels, names in val_loader:
            inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device).unsqueeze(1)
            output = model.forward(inputs)
            valloss = BCELoss(output, labels)
            val_loss += valloss.item()*inputs.size(0)
            labelarr = torch.cat((labelarr,labels.squeeze()),0)
            outputarr = torch.cat((outputarr,output.squeeze()),0)
            acc = binary_acc(output, labels)
            accuracy += acc.item()

    scheduler.step()
    outarr = outputarr.cpu().numpy()
    labarr = labelarr.cpu().numpy()
    fper, tper, thresholds = roc_curve(labarr, outarr)
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    save_name = os.path.join(args.datapath, str(epoch)+name+'.pt')
    torch.save(model,save_name)
    print('auc',auc(fper, tper))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
