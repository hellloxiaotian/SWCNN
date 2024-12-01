import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from PIL import Image


def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    print("img.shape", img.shape)
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, patch_size, stride, aug_times=1, mode='gray'):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]

    if mode == 'gray':
        files = glob.glob(os.path.join(data_path, 'train', '*.png'))
        files.sort()
        h5f = h5py.File(data_path + "/" + 'train.h5', 'w')
    elif mode == "color":
        # files = glob.glob(os.path.join(data_path, 'VOC', '*.jpg'))
        files = glob.glob(os.path.join(data_path, 'SWCNN_train_data', '*.jpg'))
        files.sort()
        h5f = h5py.File(data_path + "/" + 'SWCNN_train_color.h5', 'w')

    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = Image.open(files[i])

        h, w, c = img.shape
        # c = 3
        for k in range(len(scales)):
            if mode == 'color':
                if int(h * scales[k]) < 256 or int(w * scales[k]) < 256:
                    continue
            Img = cv2.resize(img, (int(w * scales[k]), int(h * scales[k])), interpolation=cv2.INTER_CUBIC)
            # Img = img.resize( (int(h * scales[k]), int(w * scales[k])))
            if mode =='gray':
                Img = np.expand_dims(Img[:, :, 0].copy(), 0)
            else:
                Img = np.transpose(Img, (2, 0, 1))
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3] * aug_times))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times - 1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(train_num) + "_aug_%d" % (m + 1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    if mode == 'gray':
        files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
        files.sort()
        h5f = h5py.File(data_path + "/" + 'val.h5', 'w')
    elif mode == 'color':
        files = glob.glob(os.path.join(data_path, 'VOC_test', '*.jpg'))
        files.sort()
        h5f = h5py.File(data_path + "/" + 'val_color.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        if mode == 'gray':
            img = np.expand_dims(img[:, :, 0].copy(), 0)
        else:
            img = np.transpose(img, (2, 0, 1))
        # img = Image.open(files[i])
        # img = np.expand_dims(img[:, :, 0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True, mode='gray', data_path='/media/npu/Data/jtc/data/'):
        super(Dataset, self).__init__()
        self.train = train
        self.mode = mode
        self.data_path = data_path
        if mode == 'gray':
            if self.train:
                h5f = h5py.File(self.data_path + "/" + 'train.h5', 'r')
            else:
                h5f = h5py.File(self.data_path + "/" + 'val.h5', 'r')
            self.keys = list(h5f.keys())
            random.shuffle(self.keys)
            h5f.close()
        elif mode == 'color':
            if self.train:
                h5f = h5py.File(self.data_path + "/" + 'train_color.h5', 'r')
            else:
                h5f = h5py.File(self.data_path + "/" + 'val_color.h5', 'r')
            self.keys = list(h5f.keys())
            random.shuffle(self.keys)
            h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.mode == 'color':
            if self.train:
                h5f = h5py.File(self.data_path + "/" + 'train_color.h5', 'r')
            else:
                h5f = h5py.File(self.data_path + "/" + 'val_color.h5', 'r')
        else:
            if self.train:
                h5f = h5py.File(self.data_path + "/" + 'train.h5', 'r')
            else:
                h5f = h5py.File(self.data_path + "/" + 'val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
