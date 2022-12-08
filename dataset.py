import os
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image, ImageFile
from albumentations import pytorch
from torch.utils.data import Dataset, DataLoader



#
class CustomDataset(Dataset):

    def __init__(self, root, dataset, transform):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        self.path = os.path.join(root, dataset)

        self.data = pd.read_csv( os.path.join(self.path, "data.csv") )
        self.classes = open(os.path.join(self.path, "classes.txt"), 'r').read().splitlines()
        
        self.transform = transform
        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        img_path = os.path.join(self.path, self.data.iloc[idx, 0])
        label_name = self.data.iloc[idx, 1]
        # X
        img = Image.open(img_path)
        image = img.convert('RGB') if len(img.size) == 2 else img
        image = self.transform(image = np.array(image))["image"]            
        # Y
        label = self.classes.index(label_name)
        return image, label



#
def get_valid_transforms(size):

    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tr = [
        A.Resize(size, size),
        A.Normalize(*mean_std),
        A.pytorch.transforms.ToTensorV2()
    ]

    return A.Compose(tr)



#
def get(root, dataset, transforms, batch_size, shuffle = False):
    
    dataset = CustomDataset(root = root, dataset = dataset, 
                             transform = transforms)
    
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    return loader

