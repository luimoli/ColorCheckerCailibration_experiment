import os, sys
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.path.abspath('../'))

import glob
import cv2
import numpy as np
# import cv2.cv2 as cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
# from utils import evaluate_result
from utils.misc import gamma, gamma_reverse
from utils import smv_colour

loss_list = []
blc_list = range(10)
for i in blc_list:
    analog_gt = np.float32(np.load("./data/analog/analog_gt.npy") / 1000)
    ideal_xyz = smv_colour.RGB2XYZ(torch.from_numpy(analog_gt), 'bt709').numpy()
    ideal_lab = smv_colour.XYZ2Lab(torch.from_numpy(ideal_xyz)).numpy()
    ideal_lab = np.expand_dims(np.float64(ideal_lab), 1)


    src_for_ccm = np.float32((np.load(r'data/analog/analog_D65.npy') + i) / 255.)
    rgb_gain = src_for_ccm[18].max(keepdims=True) / src_for_ccm[18]
    src_for_ccm = src_for_ccm * rgb_gain * (0.9 / src_for_ccm[18, :].max())
    print(rgb_gain)
    src_for_ccm = np.float64(src_for_ccm[:, None, :])

    model = cv2.ccm_ColorCorrectionModel(src_for_ccm, ideal_lab, cv2.ccm.COLOR_SPACE_Lab_D65_2)
    model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
    model.setCCM_TYPE(cv2.ccm.CCM_3x3)
    model.setDistance(cv2.ccm.DISTANCE_CIE2000)
    model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
    model.setLinearGamma(1)
    model.setLinearDegree(3)
    # model.setSaturatedThreshold(0, 0.98)
    model.run()
    loss_list.append(model.getLoss())
    print('loss:', model.getLoss())
    ccm = model.getCCM()
    print('ccm:\n{}\n'.format(ccm))

plt.figure()
plt.plot(blc_list, loss_list, marker='o')
plt.xlabel('add black level value')
plt.ylabel('mean deltaE00')
# plt.ylim((1.50, 1.51))
plt.show()

# mask = model.getMask()
# for i in range(len(mask)):
#     print(i, mask[i])
# print(model.getWeights())

# ccm = model.getCCM()
# print('ccm:\n{}\n'.format(ccm))
# loss = model.getLoss()
# print('loss:\n{}\n'.format(loss))
# print('ccm.sum(axis=0):', ccm.sum(axis=0))
# exit()

        