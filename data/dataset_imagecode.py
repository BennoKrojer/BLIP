import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.utils import pre_caption

# from transformers import BertTokenizerFast
from PIL import Image

class PairedImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False, max_words=40):
        super().__init__()
        self.transform = transform
        self.max_words = max_words
        image_root = '/network/scratch/b/benno.krojer/dataset/games'
        self.data = self.load_data(Path(data_dir), image_root, split, video_only)

    @staticmethod
    def load_data(data_dir, img_path, split, video_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                img_idx = int(img_idx)
                static = 'open-images' in img_dir
                target_file = img_files[img_idx]
                for i, file in enumerate(img_files):
                    if i == img_idx:
                        continue
                    if i < img_idx:
                        pair = [target_file, file]
                        label = 1
                    else:
                        pair = [file, target_file]
                        label = 0
                    if video_only:
                        if not static:
                            dataset.append((img_dir, pair, label, text))
                    else:
                        dataset.append((img_dir, pair, label, text))
        
        return dataset
    
    def __getitem__(self, idx):
        img_dir, pair, label, text = self.data[idx]
        
        images = [self.transform(Image.open(img_file).convert('RGB')) for img_file in pair]
        img1, img2 = images
        sentence = pre_caption(text, self.max_words)

        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img1, img2, sentence, label, is_video
    
    def __len__(self):
        return len(self.data)

class ImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False, max_words=40):
        super().__init__()
        self.transform = transform
        self.max_words = max_words
        image_root = '/network/scratch/b/benno.krojer/dataset/games'
        self.data = self.load_data(Path(data_dir), image_root, split, video_only)

    @staticmethod
    def load_data(data_dir, img_path, split, video_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if video_only:
                    if not static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
        
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        target_file = img_files[img_idx]

        target_img = self.transform(Image.open(target_file).convert('RGB'))

        img0 = []
        img1 = []
        labels = []

        for i, file in enumerate(img_files):
            img = self.transform(Image.open(file).convert('RGB'))
            if i == img_idx:
                continue
            if i < img_idx:
                img0.append(target_img)
                img1.append(img)
                labels.append(torch.tensor(1))
            else:
                img0.append(img)
                img1.append(target_img)
                labels.append(torch.tensor(0))
        
        img0 = torch.stack(img0)
        img1 = torch.stack(img1)
        labels = torch.stack(labels)
        sentence = pre_caption(text, self.max_words)
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img0, img1, sentence, labels, is_video, img_dir
    
    def __len__(self):
        return len(self.data)

class InferenceImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False, max_words=40):
        super().__init__()
        self.transform = transform
        self.max_words = max_words
        image_root = '/network/scratch/b/benno.krojer/dataset/games'
        self.data = self.load_data(Path(data_dir), image_root, split, video_only)

    @staticmethod
    def load_data(data_dir, img_path, split, video_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                pairs = []
                for i in range(img_files):
                    for j in range(i+1, img_files):
                        f1 = img_files[i]
                        f2 = img_files[j]
                        pairs.append((f1,f2))

                if video_only:
                    if not static:
                        dataset.append((img_dir, pairs, int(img_idx), text))
                else:
                    dataset.append((img_dir, pairs, int(img_idx), text))
        
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        target_file = img_files[img_idx]

        target_img = self.transform(Image.open(target_file).convert('RGB'))

        img0 = []
        img1 = []
        labels = []

        for i, file in enumerate(img_files):
            img = self.transform(Image.open(file).convert('RGB'))
            if i == img_idx:
                continue
            if i < img_idx:
                img0.append(target_img)
                img1.append(img)
                labels.append(torch.tensor(1))
            else:
                img0.append(img)
                img1.append(target_img)
                labels.append(torch.tensor(0))
        
        img0 = torch.stack(img0)
        img1 = torch.stack(img1)
        labels = torch.stack(labels)
        sentence = pre_caption(text, self.max_words)
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img0, img1, sentence, labels, is_video, img_dir
    
    def __len__(self):
        return len(self.data)
