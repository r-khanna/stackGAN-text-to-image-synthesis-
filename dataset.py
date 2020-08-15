
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd

#from miscc.config import cfg


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='jpg', embedding_type='embeddings1',
                 imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        # load_size = int(self.imsize * 76 / 64)
        load_size = int(self.imsize)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def load_all_captions(self):
        caption_dict = {}
        filepath = os.path.join(self.data_dir, 'caption_id.csv')
        cap=pd.read_csv(filepath)
        for key in self.filenames:
            caption_dict[key] = cap['Caption'][cap['image_id']==key]
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        embedding_filename = '/embbedings1.csv'
        f=pd.read_csv(data_dir + embedding_filename)
        embeddings=np.array(np.array(f.iloc[:,1:]))
        return embeddings

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.csv')
        filenames=np.array(pd.read_csv(filepath)['image_id'])
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        data_dir = '%s/jpg' % self.data_dir
        #captions = self.captions[key]
        embedding = self.embeddings[index,:]
        img_name = '%s/%s.jpg' % (data_dir, key)
        img = self.get_img(img_name)
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return len(self.filenames)
