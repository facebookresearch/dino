# This file contains all nifti dataset, dataloaders and samplers needed for training


import os
import numpy as np
import nibabel as nib
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler


class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None, paths_text: str = "ct_paths.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
            self.transform = transforms.Compose([
                ResizeTo512(),
                ZScoreNormalization(),
                ToPILImage(),
            ])
            self.transform = transforms.Compose([
                self.transform,
                transform
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        nifti_img = nib.load(file_name)
        try:
            data_array = nifti_img.get_fdata()

            # Handles data we know we won't need to learn
            if len(data_array.shape) != 3:
                return None

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

        # Turn into 3d.
        sampled_slice = np.broadcast_to(sampled_slice[:, :, None],
                                          (sampled_slice.shape[0], sampled_slice.shape[1], 3))

        # Apply the default transform to the sampled slice
        sampled_slice = self.transform(sampled_slice)

        sample = (sampled_slice, 0)
        return sample


class NumpyDataset(Dataset):
    """
    Dataset that uses numpy arrays of shape x,y,3 that have been sampled from DICOM files.
    """
    def __init__(self, root_dir, transform=None, paths_text: str = "ct_paths.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
            self.transform = transforms.Compose([
                ResizeTo512(),
                CTWindowing(),
                ZScoreNormalization(),
                ToPILImage(),
            ])
            self.transform = transforms.Compose([
                self.transform,
                transform
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        data_array = np.load(file_name)
        # Apply the default transform to the sampled slice
        data_array = self.transform(data_array)
        sample = (data_array, 0)
        return sample

class NumpyDatasetEval(Dataset):
    """
    Dataset that uses numpy arrays of shape x,y,3 that have been sampled from DICOM files.
    """
    def __init__(self, root_dir, transform=None, paths_text: str = "phase_no_none_test.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
            self.transform = transforms.Compose([
                ResizeTo512(),
                CTWindowing(),
                ZScoreNormalization(),
                ToPILImage(),
                transforms.ToTensor(),

            ])
            if transform:
                self.transform = transforms.Compose([
                    self.transform,
                    transform
                ])

            self.resize_and_tensor_transform = transforms.Compose([
                ResizeTo512(),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        data_array = np.load(file_name)
        display_arr = self.resize_and_tensor_transform(data_array)
        # Apply the default transform to the sampled slice
        transformed_arr = self.transform(data_array)
        label = file_name.split('/')[-3]
        sample = (transformed_arr, label, display_arr)
        return sample


class ResampleNoneSampler(Sampler):
    def __init__(self, dataset, replacement=True):
        self.dataset = dataset
        self.replacement = replacement
        self.sampler = RandomSampler(dataset)

    def __iter__(self):
        indices = list(iter(self.sampler))
        for idx in indices:
            sample = self.dataset[idx]
            while sample is None:
                if not self.replacement:
                    continue
                idx = torch.randint(len(self.dataset), size=(1,)).item()
                sample = self.dataset[idx]
            yield idx

    def __len__(self):
        return len(self.dataset)

class ZScoreNormalization:
    def __call__(self, image):
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 0.001)


class ToTensor:
    def __call__(self, image):
        return torch.from_numpy(image).float()

class ResizeTo512:
    def __call__(self, image):
        # Rescale the image to 512x512 using bilinear interpolation
        image = torch.tensor(image)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        resized_image = torch.nn.functional.interpolate(image, size=(512, 512, 3), mode='trilinear', align_corners=False)
        return resized_image[0, 0].numpy()

class ToPILImage:
    def __call__(self, array):
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(np.uint8(array))

        return pil_image

class CTWindowing:

    def __call__(self, array):
        # all values smaller than -300 and larger than 300 send to the median

        median = np.median(array)
        array[np.where(array < -300)] = median
        array[np.where(array > 300)] = median
        return array
