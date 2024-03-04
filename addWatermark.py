#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-2 16:14
# @Author  : 26731
# @File    : addWatermark.py
# @Software: PyCharm
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import random


# angle = random.randint(-90,90)
# scale = random.random()
# scale *=2
# img = Image.open("water.png")
# img = img.rotate(angle,expand = 1)
# w,h = img.size
# img = img.resize((int(w*scale),int(h*scale)))
# img_three = Image.open("three.jpg")

# out = Image.alpha_composite(img_three,img.resize(img_three.size))
# img_three.paste(img,(1000,3000),img)
# plt.imshow(img_three)
#
# plt.show()
#     img = img.copy()
#     TRANSPARENCY = random.randint(28, 82)
#
#     image = Image.fromarray(img)
#     watermark = Image.open('./水印.png')  # 水印路径
#     # cv2.imshow(watermark)
#     plt.imshow(watermark)
#     if watermark.mode != 'RGBA':
#         alpha = Image.new('L', watermark.size, 255)
#         watermark.putalpha(alpha)
#
#     random_X = random.randint(-750, 45)
#     random_Y = random.randint(-500, 30)
#
#     paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)
#     image.paste(watermark, (random_X, random_Y), mask=paste_mask)

def add_watermark_noise(noise, occupancy=50):
    watermark = Image.open("water.png")
    noise = noise.numpy()

    _, h, w = noise.shape
    img_for_cnt = np.zeros((h, w), np.uint8)
    occupancy = np.random.uniform(0, occupancy)

    # 获取噪声的shape
    noise = np.ascontiguousarray(np.transpose(noise, (1, 2, 0)))
    noise_h, noise_w, noise_c = noise.shape
    noise = np.uint8(noise*255)
    # plt.imshow(noise)
    # plt.show()
    img_for_cnt = np.zeros((noise_h, noise_w), np.uint8)
    noise = Image.fromarray(noise)
    img_for_cnt = Image.fromarray(img_for_cnt)
    w, h = watermark.size
    while True:
        # 随机选取放缩比例和旋转角度
        angle = random.randint(-45, 45)
        scale = random.random()
        # scale = scale
        # 旋转水印
        img = watermark.rotate(angle, expand=1)
        #  放缩水印
        img = img.resize((int(w * scale), int(h * scale)))
        # 将噪声转换为PIL

        # 随机选取要粘贴的部位

        x = random.randint(int(-w*scale), noise_h)
        y = random.randint(int(-h*scale), noise_w)
        noise.paste(img, (x, y), img)
        # plt.imshow(noise)
        # plt.show()

        img_for_cnt.paste(img, (x, y), img)
        # plt.imshow(img_for_cnt, cmap='gray')
        # plt.show()
        img_cnt = np.array(img_for_cnt)
        sum = (img_cnt > 0).sum()
        ratio = noise_h * noise_w * occupancy / 100
        if sum > noise_h * noise_w * occupancy / 100:
            # noise = torch.FloatTensor(noise)
            break
    return noise


if __name__ == '__main__':
    noise = torch.FloatTensor(torch.ones((128, 3, 200, 200)).size()).normal_(mean=0, std=50 / 255.)
    max_noise = torch.max(noise)
    min_noise = torch.min(noise)
    noise = (noise - min_noise)/(max_noise-min_noise)
    # noise = torch.zeros((128, 3, 200, 200))
    add_watermark_noise(noise[0])
