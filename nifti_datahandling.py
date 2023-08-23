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
                ResizeToX(512),
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
                ResizeToX(512),
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
                ResizeToX(512),
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
                ResizeToX(128),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        if file_name.endswith('npz'):
            data_array = np.load(file_name)['arr_0.npy']
        else:
            data_array = np.load(file_name)
        display_arr = self.resize_and_tensor_transform(data_array)
        # Apply the default transform to the sampled slice
        transformed_arr = self.transform(data_array)
        label = file_name.split('/')[-3]
        sample = (transformed_arr, label, display_arr)
        return sample


class NumpyDatasetAllModalities(Dataset):
    """
    Dataset that uses numpy arrays of shape x,y,3 that have been sampled from DICOM files.
    The dataset can either contain images from MRI or CT scans which need to be transformed slightly
    differently.

    Currently, the way to know which modality is used is according to the dataset path itself,
    this may change in the future.
    """
    def __init__(self, root_dir, transform=None, paths_text: str = "phase_no_none_test.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
            self.ct_transform = transforms.Compose([
                ResizeToX(512),
                CTWindowing(),
                ZScoreNormalization(),
                ToPILImage(),

            ])
            self.mri_transform = transforms.Compose([
                ResizeToX(512),
                ZScoreNormalization(),
                ToPILImage(),
            ])


            if transform:
                self.ct_transform = transforms.Compose([
                    self.ct_transform,
                    transform
                ])
                self.mri_transform = transforms.Compose([
                    self.mri_transform,
                    transform
                ])

            self.resize_and_tensor_transform = transforms.Compose([
                ResizeToX(128),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        if file_name.endswith('npz'):
            data_array = np.load(file_name)['arr_0.npy']
        else:
            data_array = np.load(file_name)
        # Apply the default transform to the sampled slice
        label = file_name.split('/')[-3]

        #TODO this is obviously a bad solution because these labels tend to change. find a better solution?
        if label == "Brain":
            transform = self.mri_transform
        elif label == "Lungs" or label == "Liver":
            transform = self.ct_transform
        else:
            raise ValueError("unclear label being used")
        transformed_arr = transform(data_array)
        sample = (transformed_arr, 0)
        return sample

class NumpyDatasetEvalAllModalities(Dataset):
    """
    Dataset that uses numpy arrays of shape x,y,3 that have been sampled from DICOM files.
    The dataset can either contain images from MRI or CT scans which need to be transformed slightly
    differently.

    Currently, the way to know which modality is used is according to the dataset path itself,
    this may change in the future.
    """
    def __init__(self, root_dir, transform=None, paths_text: str = "phase_no_none_test.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
            self.ct_transform = transforms.Compose([
                ResizeToX(512),
                CTWindowing(),
                ZScoreNormalization(),
                ToPILImage(),
                transforms.ToTensor()

            ])
            self.mri_transform = transforms.Compose([
                ResizeToX(512),
                ZScoreNormalization(),
                ToPILImage(),
                transforms.ToTensor()
            ])


            if transform:
                self.ct_transform = transforms.Compose([
                    self.ct_transform,
                    transform
                ])
                self.mri_transform = transforms.Compose([
                    self.mri_transform,
                    transform
                ])

            self.resize_and_tensor_transform = transforms.Compose([
                ResizeToX(64),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        if file_name.endswith('npz'):
            data_array = np.load(file_name)['arr_0.npy']
        else:
            data_array = np.load(file_name)
        try:
            display_arr = self.resize_and_tensor_transform(data_array)
        except Exception as e:
            print(e)
            print(f"problem file is {file_name}")
            raise e
        # Apply the default transform to the sampled slice
        label = file_name.split('/')[-3]

        #TODO this is obviously a bad solution because these labels tend to change. find a better solution?
        if label == "Brain":
            transform = self.mri_transform
        elif label == "Lungs" or label == "Liver":
            transform = self.ct_transform
        else:
            # Make sure this is fine if you reach this point!
            transform = self.ct_transform
        transformed_arr = transform(data_array)
        sample = (transformed_arr, label, display_arr)
        return sample


class NumpyDatasetMRIEmbeddings(Dataset):
    """
    Dataset that uses numpy arrays of shape x,y,3 that have been sampled from DICOM files.
    The dataset can either contain images from MRI or CT scans which need to be transformed slightly
    differently.

    Currently, the way to know which modality is used is according to the dataset path itself,
    this may change in the future.
    """
    def __init__(self, root_dir, transform=None, paths_text: str = "phase_no_none_test.txt"):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, paths_text), 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [os.path.join(self.root_dir, file.strip()) for file in self.file_list]
            self.mri_transform = transforms.Compose([
                ResizeToX(512),
                ZScoreNormalization(),
                ToPILImage(),
                transforms.ToTensor()
            ])

            if transform:
                self.mri_transform = transforms.Compose([
                    self.mri_transform,
                    transform
                    ])
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        if file_name.endswith('npz'):
            data_array = np.load(file_name)['arr_0.npy']
        else:
            data_array = np.load(file_name)

        transform = self.mri_transform
        transformed_arr = transform(data_array)
        sample = (transformed_arr, self.file_list[idx])
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

class ResizeToX:

    def __init__(self, x:int):
        self.x = x

    def __call__(self, image):
        # Rescale the image to 512x512 using bilinear interpolation
        image = torch.tensor(image)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        resized_image = torch.nn.functional.interpolate(image, size=(self.x, self.x, 3), mode='trilinear', align_corners=False)
        return resized_image[0, 0].numpy()

class ResizeTo512SameSlice:
    def __call__(self, image):
        # Rescale the image to 512x512 using bilinear interpolation
        image = torch.tensor(np.stack([image[:, :, 1]] * 3, axis=0))
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


class MRINormalization:

    def __call__(self, array):

        # find background by first getting rid of small values close to 0 that are likely to be the background

        # get rid of low value outliers
        x = np.clip(array, a_min=0, a_max=None)

        # min max norm and discard very small values
        normalized_min_max = (x - x.min()) / (x.max() - x.min() + 0.001)
        x[normalized_min_max < 0.01] = np.nan

        # quantile standardization to get rid of large outliers

        quantile_data = (x - np.nanmedian(x)) / (np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25) + 0.001)
        quantile_data[np.logical_or(-1 >= quantile_data, quantile_data >= 1)] = np.nan

        # now we can safely z-score normalize the image.
        mean = np.nanmean(quantile_data)
        std = np.nanstd(quantile_data)

        #clip data and then normalize
        x = (x - np.nanmedian(x)) / (np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25) + 0.001)
        x[-1 >= quantile_data] = -1
        x[quantile_data >= 1] = 1
        x[np.isnan(x)] = -1
        x = (x - mean) / (std + 0.0001)
        x = (x - x.min()) / (x.max() - x.min() + 0.0001)
        array = x
        return array