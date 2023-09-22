
import cv2
import torch
import numpy as np
# from numpy import genfromtxt
import matplotlib.pyplot as plt


image_grey = cv2.imread(r"data\tmp\grey.BMP")[..., ::-1] / 255.
wp_area = image_grey[704:1065, 931:1519, :].copy()
wp_value = wp_area.mean(axis=(0,1))
gain = np.max(wp_value) / wp_value
image = image_grey * gain[None, None]
image = np.clip(image, 0, 1)
cv2.imwrite(r"data\tmp\grey_new.png", image[..., ::-1] * 255.)



# image_A = cv2.imread(r"data\cc_wp\m1_A.BMP")[..., ::-1] / 255.
# image_ori = image_H.copy()

# wp_area = image_H[477:778, 973:1545, :]
# cc_area = image_H[1776:1846, 645:719, :]
# wp_area_mean = wp_area.mean()
# gain = np.max(wp_area_mean)[:, None] / wp_area_mean
# rgb_gain = gain

# for i in range(2, 256, 1):
#     for j in range(2, 256, 1):
#         image_ori = image_H.copy()
#         wp_value = np.array([i, 200, j]) / 255.
#         gain = np.max(wp_value) / wp_value

#         image_ori = image_ori * gain[None, None]
#         # image = np.clip(image, 0, 1)
#         cc_value = image_ori[1776:1846, 645:719, :].mean(axis=(0,1)) 
#         if abs(1 - (cc_value[0] / cc_value[1])) <=0.1:
#             print(cc_value)

