import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
from scipy.io import loadmat
import numpy as np

class UCF50(data.Dataset):
    def __init__(self, data_path, folder, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_D1_path = data_path + '/den1'
        self.gt_D2_path = data_path + '/den2'
        self.gt_path = data_path + '/pts'

        self.mode = mode

        self.img_files = []
        self.gt_D1_files = []
        self.gt_D2_files = []
        self.gt_files = []

        for i_folder in folder:
            folder_img = self.img_path + '/' + str(i_folder)
            folder_gt_D1 = self.gt_D1_path + '/' + str(i_folder)
            folder_gt_D2 = self.gt_D2_path + '/' + str(i_folder)
            folder_gt = self.gt_path + '/' + str(i_folder)
            for filename in os.listdir(folder_img):
                if os.path.isfile(os.path.join(folder_img,filename)):
                    self.img_files.append(folder_img + '/' + filename)
                    self.gt_files.append(folder_gt + '/' + filename.split('.')[0] + '_ann.mat')   
                    self.gt_D1_files.append(folder_gt_D1 + '/' + filename)   
                    self.gt_D2_files.append(folder_gt_D2 + '/' + filename)   

        self.num_samples = len(self.img_files) 

        self.mode = mode
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        
        
    
    def __getitem__(self, index):

        img, den1, den2, deni = self.read_image_and_gt(index)
      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.gt_transform is not None:
            den1 = self.gt_transform(den1)  
            den2 = self.gt_transform(den2)
            deni = self.gt_transform(deni)
            
        return img, den1, den2, deni
    
    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,index):
        img = Image.open(self.img_files[index])
        if img.mode == 'L':
            img = img.convert('RGB')
        
        #get points for localisation and ground truth
        points = self.getPoints(self.gt_files[index])
        gtval = len(points)
        
        # density map with k = 1
        den1 = Image.open(self.gt_D1_files[index])
        den1 = (den1/np.sum(den1))*gtval

        # density map with k = 2
        den2 = Image.open(self.gt_D2_files[index])
        den2 = (den2/np.sum(den2))*gtval
        
        # density map (localisation map) with k = inf,
        deni = self.computeLocMap( points, img.size, gtval )
        
        return img, den1, den2, deni


    def get_num_samples(self):
        return self.num_samples       
            
    def getPoints(self,filename):
        x = loadmat(filename)
        return x['annPoints']

    def computeLocMap(self, points, size, N ):
        w,h = size
        DenseMap = np.zeros((h,w))
        for i in range(N):
            xpi = points[i,0]
            ypi = points[i,1]
            x = int(min(xpi,w-1))
            y = int(max(ypi,0))
            DenseMap[y,x] = 1.0
        return DenseMap