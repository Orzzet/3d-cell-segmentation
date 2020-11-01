import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import h5py
import os
import numpy as np

PATHS = {
'LQ_NOBATCH_TRAIN' : 'data/cells/train_lq_nobatch/',
'LQ_NOBATCH_VALID' : 'data/cells/valid_lq_nobatch/',
'LQ_NOBATCH_TEST' : 'data/cells/test_lq_nobatch/',
'LQ_BATCH_TRAIN' : 'data/cells/train_lq_batch/',
'LQ_BATCH_VALID' : 'data/cells/valid_lq_batch/',
'LQ_BATCH_TEST' : 'data/cells/test_lq_batch/',
'HQ_BATCH_TEST' : 'data/cells/test_hq_batch/',
'MQ_BATCH_TEST' : 'data/cells/test_mq_batch/'
}


class CellsDataset(VisionDataset):

    def __init__(self, folder_path, target_mode='target', dim_size_reduction=(1,1,1), transform=None, transform_augmentation=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
        c, x, y, z = image.shape
        x, y, z = x//self.dim_size_reduction[0], y//self.dim_size_reduction[1], z//self.dim_size_reduction[2]
        return image[:,i*x:(i+1)*x,j*y:(j+1)*y,k*z:(k+1)*z]
    
    def __getitem__(self, index):
        current_file = h5py.File(self.files[index][0], 'r')
        correct_cell_count, resized_cell_count = current_file.attrs['correct_cell_count'], current_file.attrs['cell_count']
        image = current_file.get('image')
        #image = self.reduce_dims(image, index)
        image = np.array(image, dtype='f4')
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        if self.target_mode == 'boundaries':
            target = current_file.get('boundaries')
        elif self.target_mode == 'not_segmented_target':
            target = current_file.get('not_segmented_target')
        elif self.target_mode == 'target':
            target = current_file.get('target')
        #target = self.reduce_dims(target, index)
        target = np.array(target, dtype='i1')
        image = Variable(torch.tensor(image))
        current_file.close()
        if self.transform:
            image = self.transform(image)
        image = image.permute(0,3,1,2)
        target = np.moveaxis(target, -1, 1)
        target = Variable(torch.tensor(target))
        if self.transform_augmentation:
            both = torch.stack((image, target))
            image, target = self.transform_augmentation(both)
        return image, target, correct_cell_count, resized_cell_count

