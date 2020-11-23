import fnmatch
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import random

class BackgroundDataset(Dataset):
    def __init__(self, img_dir, imgsize, shuffle=True, max_num=99999):
        n_jpeg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpeg'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_jpeg_images + n_jpg_images
        self.len = min(n_images, max_num)
        self.img_dir = img_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.jpeg') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.img_names = self.img_names[:self.len]
        random.shuffle(self.img_names)
        self.shuffle = shuffle
        self.img_paths = []
        for i, img_name in enumerate(self.img_names):
            self.img_paths.append(os.path.join(self.img_dir, img_name))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        
        image = self.scale(image)
        transform = transforms.ToTensor()
        image = transform(image)
        return image
    
    def scale(self, img):
        w, h = img.size
        if w == h:
            scaled_img = img
        else:
            dim_to_scale = 1 if w < h else 2
            if dim_to_scale == 1:
                cropping = (h - w) / 2
                scaled_img = img.crop((0, int(cropping), w, int(cropping) + w))
            
            else:
                cropping = (w - h) / 2
                scaled_img = img.crop((int(cropping), 0, int(cropping) + h, h))
        
        resize = transforms.Resize((self.imgsize, self.imgsize))
        
        scaled_img = resize(scaled_img)  # choose here
        
        return scaled_img