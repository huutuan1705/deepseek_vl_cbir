import os
import random
import pandas as pd

from PIL import Image
from random import randint
from torch.utils.data import Dataset
from model.uitls import get_transform

random.seed(42)

class FlickrDataset(Dataset):
    def __init__(self, args, mode):
        super(FlickrDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.db_size = args.db_size
        
        self.train_data, self.test_data, self.db_data = self.split_images()
        self.train_transform = get_transform("train")
        self.transform = get_transform("other")
        
    def split_images(self):
        dir = str(self.args.root_data) + "/flickr30k.csv"
        data_csv = pd.read_csv(dir, delimiter='|')
        data_array = data_csv.to_numpy()
        
        total_images = len(data_csv)
        indices = list(range(total_images))
        random.shuffle(indices)
        
        train_indices = indices[:self.train_size]
        test_indices = indices[self.train_size:self.train_size + self.test_size]
        db_indices = indices[self.train_size + self.test_size:]
        
        train_data = data_array[train_indices]
        test_data = data_array[test_indices]
        db_data = data_array[db_indices]
        
        return train_data, test_data, db_data
    
    def __len__(self):
        if self.mode == "train":
            return self.train_size
        if self.mode == "test":
            return self.test_size
        return self.db_size
    
    def __getitem__(self, idx):
        sample = {}
        if self.mode == "train":
            # [image_name, comment_number, comment]
            image_name = self.train_data[idx][0]
            
            positive_sample = os.path.join(self.args.root_data, 'images', image_name)
            pos_img = Image.open(positive_sample).convert("RGB")
            pos_img = self.train_transform(pos_img)
            
            posible_list = list(range(len(self.train_size)))
            posible_list.remove(idx)
            negative_idx = posible_list[randint(0, len(posible_list)-1)]
            negative_name = self.train_data[negative_idx][0]
            negative_sample = os.path.join(self.args.root_data, 'images', negative_name)
            neg_img = Image.open(negative_sample).convert("RGB")
            neg_img = self.train_transform(neg_img)
            
            caption = self.train_data[idx][2]
            
            sample = {
                "pos_img": pos_img,
                "neg_img": neg_img,
                "caption": caption
            }
            return sample
        
        if self.mode == "test" or self.mode == "search":
            image_name = self.test_data[idx][0]
            positive_sample = os.path.join(self.args.root_data, 'images', image_name)
            pos_img = Image.open(positive_sample).convert("RGB")
            pos_img = self.transform(pos_img)
            
            posible_list = list(range(len(self.test_size)))
            posible_list.remove(idx)
            negative_idx = posible_list[randint(0, len(posible_list)-1)]
            negative_name = self.test_data[negative_idx][0]
            negative_sample = os.path.join(self.args.root_data, 'images', negative_name)
            neg_img = Image.open(negative_sample).convert("RGB")
            neg_img = self.transform(neg_img)
            
            caption = self.test_data[idx][2]
            
            sample = {
                "pos_img": pos_img,
                "neg_img": neg_img,
                "caption": caption
            }
            return sample
        
# def split_images(train_size, test_size):
#     data_csv = pd.read_csv("flickr30k.csv", delimiter='|')
#     data_array = data_csv.to_numpy()
    
#     total_images = len(data_csv)
#     indices = list(range(total_images))
#     random.shuffle(indices)
    
#     train_indices = indices[:train_size]
#     test_indices = indices[train_size:train_size + test_size]
#     db_indices = indices[train_size + test_size:]
    
#     train_data = data_array[train_indices]
#     test_data = data_array[test_indices]
#     db_data = data_array[db_indices]
    
#     return train_data, test_data, db_data

# train_data, test_data, db_data = split_images(100, 100)
# print(type(train_data))
# posible_list = list(range(len(train_data)))
# posible_list.remove(1)

# print(len(posible_list))