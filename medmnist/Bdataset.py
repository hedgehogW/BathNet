import os
import sys
import numpy as np
import random
import pandas as pd
import cv2
import medmnist.preprocessing as  pe
import medmnist.preprocessing2 as pe2
from pathlib import Path
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from .info import INFO
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
np.random.seed(0)

#img formate
img_type = '.jpg'


class B_dataset(Dataset):

    flag = ...

    def __init__(self,
                 img_path,
                 label_path,
                 transform=None,
                 target_transform=None):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
        '''
        # super is use to get the function of Dataset class.
        super(Dataset).__init__()
        self.transform = transform

        #Check whether the address is valid
        assert Path(label_path).exists()
        #Check if the specified file is a directory
        assert Path(img_path).is_dir()

        #read the csv file
        self.label_file = pd.read_csv(label_path)
        self.img_path  = Path(img_path)
        


        #drop the content with NanN value
        #self.label_file.dropna(axis = 0 , how = 'any' , inplace = True)
        print(self.label_file)
        
        #Handling the gender column
        #self.label_file.loc[:,('sex')]= [ 1 if i=='M' else 0 for i in self.label_file['sex'].values]
       #Different categories
        self.id_name = self.label_file.iloc[:,0].values  #img id
        #self.basic_info = self.label_file.iloc[:,2:4].values #gender and age
        #self.locals = self.label_file.iloc[:,4].values #matching area info Label Transfer
        self.locals = self.label_file.iloc[:,1].values   #huangbanshuizhong
        #self.globals = self.label_file.iloc[:,1].values #FFA img info Label Transfer
        self.target_transform = target_transform
        
        self.transformations = transforms.Compose([transforms.Resize([128,128]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
       


    def __getitem__(self, idx):

        # / is use to Splicing
        img_name_b = self.img_path / (self.id_name[idx] )
        #img_name_ffa = self.img_path / (self.id_name[idx] )    
        image = Image.open(img_name_b)
        #image = Image.open(img_name_ffa)
        #img_name_cfp = str(img_name_cfp)
        #image = cv2.imread(img_name_cfp)
        image = pe2.ExImg2(image)
        #image = pe.convert(image)
        image = np.array(image)

        #ba_in = tensor(self.basic_info[idx])
        #Adjust the output label as needed
        label = tensor(self.locals[idx])
        #label = tensor(self.globals[idx])
        
        """prob = 20  # prob set 0 to close cutmix
        if random.randint(0, 99) < prob :
            rand_index = random.randint(0, len(self.label_file) - 1) #create a random number

            rand_row = self.label_file.iloc[rand_index]  #all information gai hang
            rand_label = tensor(self.locals[rand_index])  # cut image's label
            
            img_name_cfp_rd = self.img_path / (self.id_name[rand_index] + img_type)
            rand_image = Image.open(img_name_cfp_rd)
            rand_image = pe2.ExImg(rand_image)
            image = image.astype(np.float32)
            rand_image = rand_image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rand_image = cv2.cvtColor(rand_image, cv2.COLOR_BGR2RGB)

            #rand_masks = cv2.imread(os.path.join(rand_row['mask_path'], rand_fn), cv2.IMREAD_GRAYSCALE)/255
            #rand_masks = cv2.resize(rand_masks, (128, 128), interpolation=cv2.INTER_LINEAR)

            lam = np.random.beta(1,1)
            bbx1, bby1, bbx2, bby2 = pe2.rand_bbox(image.shape, lam)

            image[bbx1:bbx2, bby1:bby2, :] = rand_image[bbx1:bbx2, bby1:bby2, :]
            #masks[bbx1:bbx2, bby1:bby2] = rand_masks[bbx1:bbx2, bby1:bby2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.shape[1] * image.shape[0]))
            label = int(label * lam + rand_label * (1. - lam))"""
            


        image = self.transform(image)
        #print("now here is 1 label:")
        #print("now here is 4 label:")
        """print(image,label)
        print(image.shape)
        print(label.shape)"""
        
        """print("this is image_Data")
        print(image)
        print("this is image_Data shape")
        print(image.shape)"""
        """print("this is label")
        print(label)
        print("this is label shape")
        print(label.shape)"""
        
        return  image,label

    def __len__(self):
        return self.id_name.shape[0]
