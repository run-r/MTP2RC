import os
import numpy as np
import cc3d

import SimpleITK as sitk
_interpolator_image = 'linear'          # interpolator image
_interpolator_label = 'linear'

def Eliminate_false_positives(volume):
    labels_out, N = cc3d.connected_components(volume, connectivity=26,return_N=True)  # free
    item_lst = []
    if N > 0:
        for segid in range(1, N + 1):
            extracted_image = labels_out * (labels_out == segid)
            item_lst.append(extracted_image.sum())
        item_lst = np.array(item_lst)
        max_item = item_lst.argsort()[-1]
        output = labels_out * (labels_out == (max_item+1))
        return output
    else:
        return None



def txt2list(path):
    save_list = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip('\n')
            save_list.append(line)
        file.close()

    return save_list



def resize_image_itk3d(itkimage, newSize, interpolator):
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    if len(originSize) == 2:
        new_Size = [int(newSize[0]), int(newSize[1])]
        new_Size = np.array(new_Size, float)
        factor = originSize / new_Size
    else:
        new_Size = [int(newSize[0]), int(newSize[1]),16]
        new_Size = np.array(new_Size, float)
        factor = originSize / new_Size


    newSpacing = originSpacing * factor
    new_Size = new_Size.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(new_Size.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(_SITK_INTERPOLATOR_DICT[interpolator])
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled




def resize_image_itk(itkimage, newSize, interpolator):
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    if len(originSize) == 2:
        new_Size = [int(newSize[0]), int(newSize[1])]
        new_Size = np.array(new_Size, float)
        factor = originSize / new_Size
    else:
        new_Size = [int(newSize[0]), int(newSize[1]),int(originSize[2])]

        new_Size = np.array(new_Size, float)
        factor = originSize / new_Size


    newSpacing = originSpacing * factor
    new_Size = new_Size.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(new_Size.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(_SITK_INTERPOLATOR_DICT[interpolator])
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled




class Resize3D(object):

    def __init__(self, new_size, check=True):
        self.name = 'Resize3D'
        self.new_size = new_size
        self.check = check

    def __call__(self, sample):
        t2w_image = sample['t2w']
        d_image = sample['d']
        adc_image = sample['adc']
        lesion_image = sample['lesion']
        k_image = sample['k']

        new_size = self.new_size
        check = self.check

        if check is True:
            t2w_image = resize_image_itk3d(t2w_image, new_size, _interpolator_image)
            d_image = resize_image_itk3d(d_image, new_size, _interpolator_image)
            adc_image = resize_image_itk3d(adc_image, new_size, _interpolator_image)
            lesion_image = resize_image_itk3d(lesion_image, new_size, _interpolator_label)
            k_image = resize_image_itk3d(k_image, new_size, _interpolator_label)

            return {'t2w': t2w_image,
                    'd': d_image,
                    'adc': adc_image,
                    'lesion': lesion_image,
                    'k': k_image}

        if check is False:
            return {'t2w': t2w_image,
                    'd': d_image,
                    'adc': adc_image,
                    'lesion': lesion_image,
                    'k': k_image}



def create_list(data_path):
    """
    this function is create the data list and the data is set as follow:
    --data                                 -- train_test
        --data_1                                   -- CaiFengXiang_2549481
            image.nii                                       -- t2w.mha
            label.nii                                       -- dwi.mha
        --data_2
            image.nii
            label.nii
        ...
    if u use your own data, u can rewrite this function
    """
    data_list = txt2list(data_path)

    t2w_name = 't2w.nii.gz'
    d_name = 'D.nii.gz'
    adc_name = 'adc.nii.gz'
    lesion_name = 't2a_Merge.nii.gz'
    K_name = 'K.nii.gz'

    data_list.sort()
    list_all = [{'t2w': os.path.join(path, t2w_name),
                 'd': os.path.join(path, d_name),
                 'adc': os.path.join(path, adc_name),
                 'lesion': os.path.join(path, lesion_name),
                 'k': os.path.join(path, K_name)} for path in data_list]

    return list_all



