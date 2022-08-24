from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image
import random
# from .transforms import functional
# random.seed(1234)
# from .transforms import functional
import cv2
from .transforms import transforms
from torchvision import transforms
import math
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur,GaussianBlur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomBrightness,ToSepia
)

class dataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file = datalist_file
        self.image_list, self.label_list, self.idx = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        image_ids = []
        # with open(self.datalist_file) as f:
        #     for line in f:
        #         info = line.strip().split()
        #         image_ids.append(int(info[0]))
        # self.image_ids = image_ids
        self.transform = transform
        self.aug = strong_aug()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, self.image_list[idx])
        img_name = self.image_list[idx]
        IMG_DIR = '/data/Dataset/CUB_200_2011/images'
        img_root = os.path.join(IMG_DIR, img_name)
        image = Image.open(img_root).convert('RGB')
        im_rot = image.rotate(30)
        #######aug
        imag = cv2.imread(img_root)
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        im_aug = self.aug(image=imag)['image']
        # im_rot = image.transpose(Image.ROTATE_180)
        # plt.figure("dog")
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # ax.imshow(image)
        # # plt.show()
        # # plt.figure("cat")
        # # plt.subplot(2, 1, 2)
        # ax1 = fig.add_subplot(1, 2, 2)
        # ax1.imshow(im_aug)
        # plt.show()
        # image = Image.fromarray(image, mode='RGB')
        im_aug = Image.fromarray(im_aug, mode='RGB')
        r = np.random.rand(1)
        if r <= 0.25 and 'train' in self.datalist_file:
            image = self.transform(im_rot)
        elif 0.25 < r < 0.5 and 'train' in self.datalist_file:
            image = self.transform(im_aug)
        else:
            image = self.transform(image)
        # if r < 0.5 and 'train' in img_root:
        #     image = self.transform(im_aug)
        # else:
        #     image = self.transform(image)
            # plt.figure("cat")
            # plt.imshow(image)
            # plt.show()

        if self.with_path:
            return img_name, image,  self.label_list[idx], self.idx[idx]
        else:
            return image, self.label_list[idx], self.idx[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        img_idx = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    idx, image, cls = line.strip().split()  # val
                    labels = int(cls)
                    # line = line.strip().split()
                    # image = line[0]
                    # labels = map(int, line[1:])
            idx = np.array(idx)
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
            img_idx.append(idx)
        return img_name_list, np.array(img_labels, dtype=np.float32), np.array(img_idx, dtype=np.float32)

def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id

class dataset_with_mask(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, mask_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        mask_name = os.path.join(self.mask_dir, get_name_id(self.image_list[idx])+'.png')
        mask = cv2.imread(mask_name)
        mask[mask == 0] = 255
        mask = mask - 1
        mask[mask == 254] = 255

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, mask, self.label_list[idx]
        else:
            return image, mask, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)

def strong_aug(p=0.5):

    return Compose([
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.25),
            GaussianBlur(p=0.5),
            Blur(blur_limit=3, p=0.25),
        ], p=0.2),
        HueSaturationValue(p=0.2),
        OneOf([
            # RandomBrightness(),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.6),
        ToSepia(p=0.1)
    ], p=p)


if __name__ == '__main__':
    # datalist = '/data/zhangxiaolin/data/INDOOR67/list/indoor67_train_img_list.txt';
    # data_dir = '/data/zhangxiaolin/data/INDOOR67/Images'
    # datalist = '/data/zhangxiaolin/data/STANFORD_DOG120/list/train.txt';
    # data_dir = '/data/zhangxiaolin/data/STANFORD_DOG120/Images'
    # datalist = '/data/zhangxiaolin/data/VOC2012/list/train_softmax.txt';
    # data_dir = '/data/zhangxiaolin/data/VOC2012'
    datalist = '../data/COCO14/list/train_onehot.txt';
    data_dir = '../data/COCO14/images'

    data = dataset(datalist, data_dir)

    img_mean = np.zeros((len(data), 3))
    img_std = np.zeros((len(data), 3))
    for idx in range(len(data)):
        img, _ = data[idx]
        numpy_img = np.array(img)
        per_img_mean = np.mean(numpy_img, axis=(0, 1))/255.0
        per_img_std = np.std(numpy_img, axis=(0, 1))/255.0

        img_mean[idx] = per_img_mean
        img_std[idx] = per_img_std

    print(np.mean(img_mean, axis=0), np.mean(img_std, axis=0))

