import torch
import h5py
import gzip
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    

    def __init__(self, image_path, label_path, num):



        # raw_labels = pd.read_csv(labels_csv)

        self.image_folder = h5py.File(image_path, 'r')
        self.images = np.array(self.image_folder['x'])[:num]
        self.label_folder = h5py.File(label_path, 'r')
        self.labels = np.array(self.label_folder['y']).squeeze()[:num]
        #self.transform = transform

        self.count = self.labels.shape[0]

    

        self.transformations = transforms.Compose([transforms.Resize(128),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):



        return self.count

    def __getitem__(self, index):
       
        image = self.images[index] #96,96,3
        label = self.labels[index]
        #image = np.resize(image, (64,64))
        image = Image.fromarray(image)
        image_data = self.transformations(image)
        """print("this is image_Data")
        print(image_data)
        print("this is image_Data shape")
        print(image_data.shape)
        print("this is label")
        print(label)
        print("this is label shape")
        print(label.shape)"""


        return image_data, label