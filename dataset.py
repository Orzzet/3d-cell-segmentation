import h5py
import os
import numpy as np
import torch
import torch.nn as nn
import torchio.transforms as transformsio
import kornia.augmentation as K
from torchvision.datasets import VisionDataset
from torch.autograd import Variable
from torchvision import transforms

PATHS = {
'RAW' : 'data/cells/1z 1x 1y/',
'LQ_TRAIN' : 'data/cells/0.25z 0.125x 0.125y/train/',
'LQ_VALID' : 'data/cells/0.25z 0.125x 0.125y/valid/',
'LQ_TEST' : 'data/cells/0.25z 0.125x 0.125y/test/',
'MQ_TRAIN' : 'data/cells/0.5z 0.25x 0.25y/train/',
'MQ_VALID' : 'data/cells/0.5z 0.25x 0.25y/valid/',
'MQ_TEST' : 'data/cells/0.5z 0.25x 0.25y/test/',
'HQ_TRAIN' : 'data/cells/0.5z 0.5x 0.5y/train/',
'HQ_VALID' : 'data/cells/0.5z 0.5x 0.5y/valid/',
'HQ_TEST' : 'data/cells/0.5z 0.5x 0.5y/test/',
}


class CellsDataset(VisionDataset):

    def __init__(self, folder_path, target_mode='target', dim_size_reduction=(1,1,1), transform=None, transform_augmentation=None):
        """
        Args:
            dim_size_reduction (Tuple[Int, Int, Int], optional): patches size, with (1,1,1) there will be one patch with the same size of the original
                image, with (0.5, 0.5, 0.5) there will be 8 patches, each patch with half the size on each dimension.
            transform (callable, optional): Optional transform to be applied on a original image.
            transform_augmentation (callable, optional): Optional transform to be applied on a both original and target image.
        """
        self.files = [(folder_path + filename, (i,j,k)) for filename in list(os.walk(folder_path))[0][2]
                      for i in range(dim_size_reduction[0]) for j in range(dim_size_reduction[1]) for k in range(dim_size_reduction[2])]
        self.transform = transform
        self.transform_augmentation = transform_augmentation
        self.dim_size_reduction = dim_size_reduction
        self.target_mode = target_mode

    def __len__(self):
        return len(self.files)

    def reduce_dims(self, image, index):
        i, j, k = self.files[index][1]
        z, x, y = image.shape
        z, x, y = z//self.dim_size_reduction[2], x//self.dim_size_reduction[0], y//self.dim_size_reduction[1]
        return image[k*z:(k+1)*z,i*x:(i+1)*x,j*y:(j+1)*y]
    
    def __getitem__(self, index):
        # Load file with all images
        current_file = h5py.File(self.files[index][0], 'r')
        # Get cell count for original and for 
        correct_cell_count, resized_cell_count = current_file.attrs['correct_cell_count'], current_file.attrs['cell_count']
        image = current_file.get('image')
        image = self.reduce_dims(image, index)
        image = np.array(image, dtype='f4')
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        target = current_file.get(self.target_mode)
        target = self.reduce_dims(target, index)
        target = np.array(target, dtype='i2')
        image = Variable(torch.tensor(image))
        # Image is originally (z,x,y), we need to convert it to (c=1,x,y,z) to use the torchio transforms, where c is number of channels
        image = image.permute(1,2,0).unsqueeze(0)
        current_file.close()
        if self.transform:
            image = self.transform(image)
        # We need to convert back from (1,x,y,z) to (1,z,x,y) to use kornia data augmentation properly
        image = image.permute(0,3,1,2)
        target = Variable(torch.tensor(target))
        target = target.unsqueeze(0)
        if self.transform_augmentation:
            both = torch.stack((image, target))
            image, target = self.transform_augmentation(both)
        return image, target, correct_cell_count, resized_cell_count


def get_train_valid_dataloaders(dim_size_reduction = (1,1,1), target_mode = 'target', train_path = PATHS["LQ_TRAIN"], valid_path = PATHS["LQ_VALID"]):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 1

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([transformsio.ZNormalization()])
    transform_augmentation = nn.Sequential(
        K.RandomDepthicalFlip3D(same_on_batch=True), 
        K.RandomHorizontalFlip3D(same_on_batch=True), 
        K.RandomVerticalFlip3D(same_on_batch=True),
    #    K.RandomRotation3D((0.5, 3, 3), same_on_batch=True)
    )

    # training and test datasets
    train_data = CellsDataset(train_path, target_mode=target_mode, transform = transform, transform_augmentation=transform_augmentation, dim_size_reduction=dim_size_reduction)
    valid_data = CellsDataset(valid_path, target_mode=target_mode, transform = transform, dim_size_reduction=dim_size_reduction)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader


def get_test_dataset(target_mode = 'target', dim_size_reduction = (1,1,1), test_path = PATHS["LQ_TEST"]):
    transform = transforms.Compose([transformsio.ZNormalization()])

    test_data = CellsDataset(test_path, target_mode=target_mode, transform = transform, dim_size_reduction=dim_size_reduction)

    return test_data

