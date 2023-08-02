# This file contains all nifti dataset, dataloaders and samplers needed for training


import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None, paths_text: str = "clean_files.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
        if transform is None:
            self.transform = transforms.Compose([
                ResizeTo512(),
                ZScoreNormalization(),
                ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        nifti_img = nib.load(file_name)
        try:
            data_array = nifti_img.get_fdata()

            # Handles data we know we won't need to learn
            if len(data_array.shape) != 3:
                print(f'shape of the image is {data_array.shape}')
                return None
            else:
                print(f'yay {data_array.shape} is fine')

        except Exception as e:
            print(f'problem with {file_name}')
            raise e

        # Compute Gaussian distribution centered at the middle slice
        num_slices = data_array.shape[-1]
        middle_slice = num_slices // 2
        sigma = 0.1 * num_slices  # Adjust sigma to control the spread of the distribution
        slice_idx = int(np.clip(np.random.normal(middle_slice, sigma), 0, num_slices - 1))

        # Extract the sampled slice from the z-axis
        sampled_slice = data_array[..., slice_idx].squeeze()

        # Apply the default transform to the sampled slice
        sampled_slice = self.transform(sampled_slice)

        # Turn into 3d.
        sampled_slice = np.broadcast_to(sampled_slice[:, :, None],
                                          (sampled_slice.shape[0], sampled_slice.shape[1], 3))
        sample = (sampled_slice, 0)

        return sample


class ZScoreNormalization:
    def __call__(self, image):
        mean = image.mean()
        std = image.std()
        return (image - mean) / std


class ToTensor:
    def __call__(self, image):
        return torch.from_numpy(image).float()

class ResizeTo512:
    def __call__(self, image):
        # Rescale the image to 512x512 using bilinear interpolation
        image = torch.tensor(image)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        resized_image = torch.nn.functional.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
        return resized_image[0, 0].numpy()
