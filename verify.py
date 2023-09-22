import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rawpy
import colour

# from utils import smv_colour


class Test:
    def __init__(self) -> None:
        self.__a = 5
        self.b = 6
        self.c = 7
    
    # @property
    # def a()


    def set_a(self, value):
        """_summary_

        Args:
            value (_type_): _description_
        """
        self.__a = value

tst = Test()
print(tst.__dict__)

# print(tst.b)
# tst.__a = 10
tst.set_a(0)
print(tst.__dict__)
    
a = np.array([[1,2,3,4,5],[1,2,3,4,5]])
# print(np.ones(2,3))
# print(a.shape)
# print(a.shape[0])




if __name__ == '__main__':
    # lab_x = np.float32(genfromtxt("./data/real_lab_xrite.csv", delimiter=',')) # from X-rite
    # lab_i = np.float32(genfromtxt("./data/real_lab.csv", delimiter=',')) # from imatest
    # xyz_x = smv_colour.Lab2XYZ(torch.from_numpy(lab_x))
    # xyz_i = smv_colour.Lab2XYZ(torch.from_numpy(lab_i))
    # xyz_x_trans = torch.einsum('ic,hc->hi', D50_TO_D65_matrix, xyz_x)
    # print(abs(xyz_x - xyz_i))
    # print(abs(xyz_x_trans - xyz_i))
    # print(xyz_x_trans)


    # shape = (2160, 3840)
    # img_r = np.ones(shape)*255
    # img_g = np.ones(shape)*255
    # img_b = np.ones(shape)*0
    # img = np.concatenate([img_r[...,None], img_g[...,None], img_b[...,None]], axis=-1)
    # plt.figure()
    # plt.imshow(img / 255.)
    # plt.show()


    # # -----verify sRGB gamma function
    a = np.ones((4,4,3))*0.5
    # res1 = gamma(a, colorspace='sRGB')
    res1_1 = a ** (1/2.2)
    # res2 = gamma_reverse(a, colorspace='sRGB')
    res2_2 = a ** 2.2
    # print('.')



    # raw=rawpy.imread(r"./data/tmp/IMG_1548.DNG")
    # img=raw.raw_image
    # img_r=img[0::2,0::2].astype('float32')
    # img_gr=img[0::2,1::2].astype('float32')
    # img_gb=img[1::2,0::2].astype('float32')
    # img_b=img[1::2,1::2].astype('float32')

    # img_g=(img_gr+img_gb)/2

    # black_level=None
    # white_level=None
    # #OB
    # if black_level is None: 
    #     black_level=raw.black_level_per_channel
    #     # black_level = 2
    # if white_level is None:
    #     # white_level = 255
    #     white_level=raw.white_level

    # img_0=(np.dstack((img_r,img_g,img_b)))/(white_level)

    # img_r=(img_r-black_level[0])/(white_level-black_level[0])
    # img_gr=(img_gr-black_level[1])/(white_level-black_level[1])
    # img_gb=(img_gb-black_level[2])/(white_level-black_level[2])
    # img_b=(img_b-black_level[3])/(white_level-black_level[3])

    # img_g=(img_gr+img_gb)/2

    # img=np.dstack((img_r,img_g,img_b))

    # img_1=img

