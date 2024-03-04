#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-7 16:54
# @Author  : 26731
# @File    : train_tri.py
# @Software: PyCharm
import os
import argparse
import string
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import HN, DnCNN, IRCNN, DPAUNet, DPUNet, SUNet, FFDNet, HN, DnCNN_RL, \
    HN2
from dataset import prepare_data, Dataset
from utils import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description="SWCNN")
config = get_config('configs/config.yaml')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers(DnCNN)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--alpha", type=float, default=0.3, help="The opacity of the watermark")
parser.add_argument("--outf", type=str, default=config['train_model_out_path_DPAUNet'], help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--mode_wm", type=str, default="S", help='with known alpha level (S) or blind training (B)')
parser.add_argument("--net", type=str, default="DPAUNet", help='Network used in training')
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=bool, default="True", help='Whether to use perception network')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--GPU_id", type=str, default="2", help='GPU_id')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_id

if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"
if opt.mode == "S":
    model_name_4 = "S" + str(opt.noiseL)
else:
    model_name_4 = "B"
if opt.loss == "L2":
    model_name_5 = "L2"
else:
    model_name_5 = "L1"
if opt.mode_wm == "S":
    model_name_6 = "aS"
else:
    model_name_6 = "aB"
tensorboard_name = opt.net + model_name_3 + model_name_4 + model_name_5 + model_name_6 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"
print()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, mode='color', data_path=config['train_data_path'])
    dataset_val = Dataset(train=False, mode='color', data_path=config['train_data_path'])
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)  # 4
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # load network
    if opt.net == "HN":
        net = HN()
    elif opt.net == "SUNet":
        net = SUNet()
    elif opt.net == "DPUNet":
        net = DPUNet()
    elif opt.net == "DPAUNet":
        net = DPAUNet()
    elif opt.net == "HN2":
        net = HN2()
    elif opt.net == "DnCNN":
        net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    elif opt.net == "DnCNN_RL":
        net = DnCNN_RL(channels=3, num_of_layers=opt.num_of_layers)
    elif opt.net == "FFDNet":
        net = FFDNet(False)
    elif opt.net == "IRCNN":
        net = IRCNN(in_nc=3, out_nc=3)
    else:
        assert False
    # TensorBoard was used to visually record the training results
    writer = SummaryWriter("runs/" + tensorboard_name)
    # 这个网络提取特征图，计算感知损失
    model_vgg = load_froze_vgg16()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    # load loss function
    if opt.loss == "L2":
        criterion = nn.MSELoss(size_average=False)
    else:
        criterion = nn.L1Loss(size_average=False)

    # criterion = nn.L1Loss(size_average=False)
    # criterion_MSE = nn.MSELoss(size_average=False)
    # Move to GPU

    # Load the trained network and continue training
    # model.load_state_dict(torch.load(os.path.join(opt.outf, "sec2_n2n_per_IRCNN_noise0.pth")))
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    step = 0
    noiseL_B = [0, 55]  #

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            random_img = random.randint(1, 12)
            if opt.mode_wm == "S":
                imgn_train = add_watermark_noise(img_train, 40, True, random_img, alpha=opt.alpha)
            else:
                imgn_train = add_watermark_noise_B(img_train, 40, True, random_img, alpha=opt.alpha)
            if opt.self_supervised == "True":
                if opt.mode_wm == "S":
                    imgn_train_2 = add_watermark_noise(img_train, 40, True, random_img, alpha=opt.alpha)
                else:
                    imgn_train_2 = add_watermark_noise_B(img_train, 40, True, random_img, alpha=opt.alpha)
            else:
                imgn_train_2 = img_train

            imgn_train_mid = torch.Tensor(imgn_train)

            if opt.mode == 'S':
                if opt.noiseL != 0:
                    noise_gauss = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)

            else:
                noise_gauss = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise_gauss.size()[0])
                for n in range(noise_gauss.size()[0]):
                    sizeN = noise_gauss[0, :, :, :].size()
                    noise_gauss[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
            if opt.noiseL == 0:
                imgn_train = torch.Tensor(imgn_train)
            else:
                imgn_train = torch.Tensor(imgn_train) + noise_gauss
            imgn_train_2 = torch.Tensor(imgn_train_2)
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            imgn_train_mid = Variable(imgn_train_mid.cuda())
            imgn_train_2 = Variable(imgn_train_2.cuda())
            if opt.net == "FFDNet":
                noise_sigma = opt.noiseL / 255.
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(img_train.shape[0])]))
                noise_sigma = Variable(noise_sigma)
                noise_sigma = noise_sigma.cuda()
                out_train = model(imgn_train, noise_sigma)
            elif opt.net == "SUNet":
                out_dn, out_train = model(imgn_train)
            elif opt.net == "DPUNet" or opt.net == "DPAUNet":
                out_train, out_dn, out_wm = model(imgn_train)
                feature_out_wm = model_vgg(out_wm)
            else:
                out_train = model(imgn_train)
            feature_out = model_vgg(out_train)

            feature_img = model_vgg(imgn_train_2)

            if opt.net == "HN":
                loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                    0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
            elif opt.net == "SUNet":
                loss = (1.0 * criterion(out_train, img_train) / imgn_train.size()[
                    0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2)) + (
                               1.0 * criterion(out_dn, imgn_train_mid) / imgn_train.size()[
                           0] * 2)
            elif opt.net == "DPUNet" or opt.net == "DPAUNet":
                loss = (1.0 * criterion(out_train, img_train) / imgn_train.size()[
                    0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2)) + (
                               1.0 * criterion(out_dn, imgn_train_mid) / imgn_train.size()[
                           0] * 2) + (
                               1.0 * criterion(out_wm, img_train) / imgn_train.size()[
                           0] * 2) + (0.024 * criterion(feature_out_wm, feature_img) / (feature_img.size()[0] / 2))
            else:
                loss = (1.0 * criterion(out_train, img_train) / imgn_train.size()[
                    0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            if opt.net == "FFDNet":
                out_train = torch.clamp(model(imgn_train, noise_sigma), 0., 1.)
            elif opt.net == "SUNet":
                out_train = torch.clamp(model(imgn_train)[1], 0., 1.)
            elif opt.net == "DPUNet" or opt.net == "DPAUNet":
                out_train = torch.clamp(model(imgn_train)[0], 0., 1.)
            else:
                out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            step += 1
            if step % 10 == 0:
                writer.add_scalar("PSNR", psnr_train, step)
                writer.add_scalar("loss", loss.item(), step)

        ## the end of each epoch
        model.eval()
        # Save the trained network parameters
        torch.save(model.state_dict(), os.path.join(opt.outf, model_name))
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                # Cut the picture into multiples of 32
                _, _, w, h = img_val.shape
                w = int(int(w / 32) * 32)
                h = int(int(h / 32) * 32)
                img_val = img_val[:, :, 0:w, 0:h]
                noise_gauss = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
                imgn_val = add_watermark_noise(img_val, 0, alpha=opt.alpha)
                img_val = torch.Tensor(img_val)
                if opt.noiseL == 0:
                    imgn_val = torch.Tensor(imgn_val)
                else:
                    imgn_val = torch.Tensor(imgn_val) + noise_gauss
                img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
                if opt.net == "FFDNet":
                    noise_sigma = opt.noiseL / 255.
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(img_val.shape[0])]))
                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.cuda()
                    out_val = torch.clamp(model(imgn_val, noise_sigma), 0., 1.)
                elif opt.net == "SUNet":
                    out_val = torch.clamp(model(imgn_val)[1], 0., 1.)
                elif opt.net == "DPUNet" or opt.net == "DPAUNet":
                    out_val = torch.clamp(model(imgn_val)[0], 0., 1.)
                else:
                    out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            writer.add_scalar("PSNR_val", psnr_val, epoch + 1)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
    writer.close()


if __name__ == "__main__":
    # data preprocess
    if opt.preprocess:
        prepare_data(data_path=config['train_data_path'], patch_size=256, stride=128, aug_times=1, mode='color')
    main()
