import time
import numpy as np
import os
import h5py
import sys
from skimage import segmentation, measure
from scipy.ndimage import zoom

def multi_instance_semseg(target):
    boundaries = segmentation.find_boundaries(target, connectivity=3, mode='outer')
    target_segmentation = np.copy(boundaries) * 1
    boundaries = (target_segmentation * -1) + 1
    target *= boundaries
    target = np.ceil(target/1000).astype('i1')
    return target, measure.label(target, return_num=True)[1], target_segmentation


def get_highest_xyz_after_reduction(filenames, original_dataset_path, dims_reduction=(1,1,1)):
    file_paths = [original_dataset_path + filename for filename in filenames]
    highest_xyz = [-1,-1,-1]
    for filename in filenames:
        with h5py.File(original_dataset_path + filename, 'r') as fr:
            image = fr.get('imageSequenceInterpolated')
            z, x, y = image.shape
            x = (x * dims_reduction[0]) + 2
            y = (y * dims_reduction[1]) + 2
            z = (z * dims_reduction[2]) + 2
            if x > highest_xyz[0]:
                highest_xyz[0] = int(x)
            if y > highest_xyz[1]:
                highest_xyz[1] = int(y)
            if z > highest_xyz[2]:
                highest_xyz[2] = int(z)
    return highest_xyz


def print_sizes(filenames, original_dataset_path):
    file_paths = [original_dataset_path + filename for filename in filenames]
    highest_xyz = [-1,-1,-1]
    for filename in filenames:
        with h5py.File(original_dataset_path + filename, 'r') as fr:
            image = fr.get('imageSequenceInterpolated')
            print(image.shape)

def store_correct_cell_count(filenames, new_dataset_path):
    file_paths = [new_dataset_path + filename for filename in filenames]
    for filename in filenames:
        with h5py.File(new_dataset_path + filename, 'a') as f:
            start = time.time()
            target_og = np.array(f.get('labelledImage3D'), dtype='i1')
            target_og = np.moveaxis(target_og, 0, -1)
            correct_num_instances = multi_instance_semseg(target_og)[1]
            f.attrs.create('correct_cell_count', [correct_num_instances])
            print("Células original: {}.".format(correct_num_instances))

def generate_preprocessed_files(filenames, compression_level, original_dataset_path, new_dataset_path, dims_reduction=(1,1,1), delete=False, batch=False):
    file_paths = [original_dataset_path + filename for filename in filenames]
    max_x, max_z, max_y = get_highest_xyz_after_reduction(filenames, dim_reduction, original_dataset_path)
    for filename in filenames:
        with h5py.File(original_dataset_path + filename, 'r') as fr:
            newfile_name = "c{}{}-{}".format(compression_level, dims_reduction, filename)
            if not os.path.exists(new_dataset_path):
                os.makedirs(new_dataset_path)
            with h5py.File("{}{}".format(new_dataset_path, newfile_name), 'a') as fw:
                if 'image' not in fw.keys():
                    start = time.time()
                    image = np.array(fr.get('imageSequenceInterpolated'), dtype='f4')
                    image = zoom(image, dims_reduction)
                    if batch:
                        z, x, y = image.shape
                        temp = np.zeros((max_z, max_x, max_y), dtype='f4')
                        temp[:z,:x,:y] = image
                        image = temp
                    print("Dimensiones image: {}".format(image.shape))
                    fw.create_dataset('image', shape=image.shape, dtype='f4', data=image, compression="gzip", compression_opts=compression_level)
                    print("Image {} {:.0f}s".format(newfile_name, time.time() - start))
                if 'target' not in fw.keys():
                    start = time.time()
                    labelledImage = fr.get('labelledImage3D')
                    correct_num_instances = fr.attrs['correct_cell_count']
                    target = np.array(labelledImage, dtype='i2')
                    target = zoom(target, dims_reduction)
                    if batch:
                        z, x, y = target.shape
                        temp = np.zeros((max_z, max_x, max_y), dtype='i2')
                        temp[:z,:x,:y] = target
                    fw.create_dataset('multi_instance_target', shape=target.shape, dtype='i2', data=target, compression="gzip", compression_opts=compression_level)
                    fw.create_dataset('not_segmented_target', shape=target.shape, dtype='i2', data=np.ceil(target/1000).astype('i2'), compression="gzip", compression_opts=compression_level)
                    if batch:
                        target = temp
                    target, num_instances, boundaries = multi_instance_semseg(target)
                    print("Dimensiones target: {}".format(target.shape))
                    print("Células original: {}. Células ahora: {}".format(correct_num_instances, num_instances))
                    fw.create_dataset('boundaries', shape=boundaries.shape, dtype='i2', data=boundaries, compression="gzip", compression_opts=compression_level)
                    fw.create_dataset('target', shape=target.shape, dtype='i2', data=target, compression="gzip", compression_opts=compression_level)
                    fw.attrs.create('correct_cell_count', correct_num_instances)
                    fw.attrs.create('cell_count', [num_instances])
                    print("Target {} {:.0f}s".format(newfile_name, time.time() - start))
        print()
        if delete:
            os.remove(original_dataset_path + filename)