import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.utils import pre_caption

# from transformers import BertTokenizerFast
from PIL import Image

class ImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False):
        super().__init__()
        self.transform = transform
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
                target_file = img_files[img_idx]
                for i, file in enumerate(img_files):
                    if i == img_idx:
                        continue
                    if i < target_file:
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
        sentence = pre_caption(text, 40)

        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img1, img2, sentence, text, label
    
    def __len__(self):
        return len(self.data)
