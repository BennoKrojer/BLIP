import os
import json
from pathlib import Path
from functools import partial
from tqdm import tqdm
from numpy import random


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizerFast
from PIL import Image

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    

class ClevrChangeDataset(Dataset):

    def __init__(self, data_dir, split, config, image_transform=None, text_transform=None):
        super().__init__()
        assert split in ['train', 'val']

        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])
        

        self.data = self.load_data(data_dir, split)

    def load_data(self, data_dir, split):
        split_file = os.path.join(data_dir, "annotations", f'{split}.json')
        with open(split_file) as f:
            json_file = json.load(f)

        dataset = []
        for i, row in tqdm(enumerate(json_file), total=len(json_file)):
            img_id = row["img_id"]
            text = row["sentences"]
            text = ". ".join(text)
            # get two different images
            image0_file = os.path.join(data_dir, "resized_images", img_id+".png")
            image1_file = os.path.join(data_dir, "resized_images", img_id+"_2.png")
            image_0 = self.image_transform(Image.open(image0_file).convert('RGB'))
            image_1 = self.image_transform(Image.open(image1_file).convert('RGB'))
            dataset.append((image_0, image_1, text))
            # if i > 500:
            #     break
        
        return dataset
    
    def __getitem__(self, idx):
        image_0, image_1, text = self.data[idx]
        
        return image_0, image_1, text
    
    def __len__(self):
        return len(self.data)

class ClevrChangeClassificationDataset(Dataset):

    def __init__(self, data_dir, split, config, image_transform=None, text_transform=None):
        super().__init__()
        assert split in ['train', 'val']

        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])

        self.captions =  self.load_captions(data_dir)
        self.data = self.load_data(data_dir, split)

    def load_captions(self, data_dir):
        fname = os.path.join(data_dir, "data", "change_captions.json")
        with open(fname) as f:
            captions = json.load(f)
        return captions


    def load_data(self, data_dir, split):
        split_file = os.path.join(data_dir, "data", 'splits.json')
        with open(split_file) as f:
            json_file = json.load(f)
        
        data = json_file[split]

        dataset = []
        for i, img_id in tqdm(enumerate(data), total=len(data)):
            text = self.captions[f"CLEVR_default_{img_id:06d}.png"][-1]
            # get two different images
            image0_file = os.path.join(data_dir, "data", "nsc_images", f"CLEVR_nonsemantic_{img_id:06d}.png")
            image1_file = os.path.join(data_dir, "data", "sc_images", f"CLEVR_semantic_{img_id:06d}.png")
            # print(text, image0_file, image1_file)
            image_0 = self.image_transform(Image.open(image0_file).convert('RGB'))
            image_1 = self.image_transform(Image.open(image1_file).convert('RGB'))
            if random.rand() > 0.5:
                target = 1
                images = [image_0,image_1]
            else:
                target = 0
                images = [image_1,image_0]
            
            img = torch.stack(images, dim=0)
            dataset.append((img, text, target))
            # if i > 50:
            #     break
        

        return dataset
    
    def __getitem__(self, idx):
        return self.data[idx] 
    
    def __len__(self):
        return len(self.data)
