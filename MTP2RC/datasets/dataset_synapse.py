import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage
import scipy
import numpy
import torchio as tio



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        res,x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def Normalization(image):
    """
    Normalize an image to 0 - 255 (8bits)
    """
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255

    return image



class TorchIOAugmentation:
    def __init__(self):
        self.transform = tio.Compose([
         
            tio.RandomAffine(scales=1.0, degrees=15, translation=0),
          
            tio.RandomAffine(scales=(0.9, 1.1), degrees=0, translation=0),
            
            tio.RandomAffine(scales=1.0, degrees=0, translation=(10, 10, 10)),
         
            tio.RandomBlur(std=(0.5, 1.5)),
        ])

    def __call__(self, sample):
        #  TorchIO Subject
        subject = tio.Subject(
            t2w=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sample['t2w']).astype(np.float32), affine=np.eye(4)),
            d=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sample['d']).astype(np.float32), affine=np.eye(4)),
            adc=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sample['adc']).astype(np.float32), affine=np.eye(4)),
            lesion=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sample['lesion']).astype(np.float32), affine=np.eye(4)),
            k=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sample['seg']).astype(np.float32), affine=np.eye(4)),
        )

        augmented = self.transform(subject)

        # 重新转换为 SimpleITK 格式
        def to_sitk(image):
            arr = image.tensor.numpy()
            arr = arr.squeeze()  # remove channel dim
            return sitk.GetImageFromArray(arr)

        return {
            't2w': to_sitk(augmented.t2w),
            'd': to_sitk(augmented.d),
            'adc': to_sitk(augmented.adc),
            'lesion': to_sitk(augmented.lesion),
            'k': to_sitk(augmented.k),
        }


class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, data_list, train=True, transform=None,clinical = None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.path = data_list
        self.bit = sitk.sitkFloat32
        self.transform = transform
        self.count = 0
        self.clinical = clinical

        cat_cols = clinical[4]
        X_mask = clinical[1]
        self.y = clinical[2]
        self.DFS = clinical[3]
        X = clinical[0]
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))

        mean = clinical[5]
        std = clinical[6]
        names = clinical[7]


        self.X1 = X[:,cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32)  # numerical columns
        self.X2 = (self.X2 - mean) / std

        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64)  # numerical columns


        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)

        self.name_dic = {}
        for item in range(len(names)):
            self.name_dic[str(names[item])] = [np.concatenate((np.array([self.cls[item]]), self.X1[item])), self.X2[item],
                                          self.y[item], self.DFS[item], np.concatenate(
                    (np.array([self.cls_mask[item]]), self.X1_mask[item])), self.X2_mask[item]]



    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image


    def __len__(self):
        return len(self.path)


    def __getitem__(self, item):

        self.count += 1
        data_dict = self.path[item]
        t2w_path = data_dict["t2w"]
        d_path = data_dict["d"]
        adc_path = data_dict["adc"]
        lesion_path = data_dict["lesion"]
        k_path = data_dict["k"]
        name = t2w_path.strip().split('/')[-1].split('\\')[0]


        # read image and label
        t2w_image = self.read_image(t2w_path)
        d_image = self.read_image(d_path)
        adc_image = self.read_image(adc_path)
        lesion_image = self.read_image(lesion_path)
        k_image = self.read_image(k_path)

        Direction = t2w_image.GetDirection()
        Origin = t2w_image.GetOrigin()
        Spacing = t2w_image.GetSpacing()
        Size = t2w_image.GetSize()

        t2w_image = Normalization(t2w_image)
        d_image = Normalization(d_image)
        adc_image = Normalization(adc_image)
        k_image = Normalization(k_image)


        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(self.bit)
        t2w_image = castImageFilter.Execute(t2w_image)
        d_image = castImageFilter.Execute(d_image)
        adc_image = castImageFilter.Execute(adc_image)
        lesion_image = castImageFilter.Execute(lesion_image)
        k_image = castImageFilter.Execute(k_image)


        sample = {'t2w': t2w_image,
                  'd': d_image,
                  'adc': adc_image,
                  'lesion': lesion_image,
                  'seg': k_image,
                  }


        for transform in self.transform:
            sample = transform(sample)


        t2w_np = sitk.GetArrayFromImage(sample['t2w'])
        d_np = sitk.GetArrayFromImage(sample['d'])
        adc_np = sitk.GetArrayFromImage(sample['adc'])
        lesion_np = sitk.GetArrayFromImage(sample['lesion'])
        k_np = sitk.GetArrayFromImage(sample['k'])


        lesion_np = abs(np.around(lesion_np))


        t2w_np = np.transpose(t2w_np, (2, 1, 0))
        d_np = np.transpose(d_np, (2, 1, 0))
        adc_np = np.transpose(adc_np, (2, 1, 0))
        lesion_np = np.transpose(lesion_np, (2, 1, 0))
        k_np = np.transpose(k_np, (2, 1, 0))


        t2w_np = t2w_np[np.newaxis, :, :, :]
        d_np = d_np[np.newaxis, :, :, :]
        adc_np = adc_np[np.newaxis, :, :, :]
        lesion_np = lesion_np[np.newaxis, :, :, :]
        k_np = k_np[np.newaxis, :, :, :]


        information_list = [Direction, Origin, Spacing,name,Size]


        if self.clinical:
            clinical_data = self.name_dic[name]
            return [torch.from_numpy(np.concatenate((t2w_np, d_np, adc_np,k_np), axis=0)),clinical_data], \
                  information_list
        else:

            return torch.from_numpy(np.concatenate((t2w_np, d_np, adc_np,k_np), axis=0)), \
                   information_list





class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(base_dir).readlines()

        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "Relabeled_train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = slice_name
            data = np.load(data_path,allow_pickle=True)
            image = data[:4,:,:]
            label = data[4,:,:]

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = vol_name+'.'
            data = np.load(filepath,allow_pickle= True)

            image = data[:4,:,:]
            label = data[4,:,:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
