import torch
import cv2
import numpy as np
import os
import torch
from numpy import genfromtxt
from utils import smv_colour

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def deltaE76(Lab1, Lab2):
    Lab1 = Lab1.cpu().numpy()
    Lab2 = Lab2.cpu().numpy()
    d_E = np.linalg.norm(Lab1 - Lab2, axis=-1)
    return d_E


def conversion(img_rgb, ccm):
    """[apply color-correction-matrix to img_rgb, and get the srgb res.]
    Args:
        img_rgb ([array]): [shape:(h,w,c=3)]
        ccm ([array]): [shape:(c,c)]
    Returns:
        [img_new_rgb]: [with gamma2.2, mantain the saturated pixels to be 1.]
    """
    # # apply the CCM
    h, w, c = img_rgb.shape
    img_rgb_reshape = np.reshape(img_rgb,(h*w, c))
    # ccres = img_rgb_reshape.mm(ccm)
    ccres = np.matmul(img_rgb_reshape, ccm)
    img_d_rgb = np.reshape(ccres, (h,w,c))
    img_d_rgb[img_d_rgb > 1] = 1
    img_d_rgb[img_d_rgb < 0] = 0

    # # apply sRGB gamma(2.2)
    # img_d_rgb = gamma(img_d_rgb,colorspace='sRGB')
    img_d_rgb = img_d_rgb ** (1 / 2.2)

    # # mantain the saturated in img_rgb
    img_d_rgb[img_rgb == 1] = 1
    return img_d_rgb

def img_convert_wccm(image_path, ccm_matrix):
    """[convert img with ccm using function 'conversion']
    Args:
        image_path ([str]): [description]
        ccm_matrix ([arr]): [description]
    """
    image_bgr = cv2.imread(image_path)  / 255.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    # image_rgb = torch.from_numpy(image_rgb)
    result_rgb_image = conversion(image_rgb, ccm_matrix)
    # result_rgb_image = result_rgb_image.cpu().numpy()
    cv2.imwrite(image_path[:-4]+'_' + 'ccm_minimize_2_test'+'.jpg', (result_rgb_image[..., ::-1] * 255.))

if __name__ == '__main__':
    image_path = r"./img/colorchecker2.jpg"




