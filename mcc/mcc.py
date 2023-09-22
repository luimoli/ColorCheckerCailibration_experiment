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

def compute_rgb_gain_from_colorchecker(mean_value):
    assert mean_value.max() <= 1, "image range should be in [0, 1]"
    # mean_value = self.calculate_colorchecker_value(image, sorted_centroid, 50)
    gain = np.max(mean_value[18:], axis=1)[:, None] / mean_value[18:, ]
    rgb_gain = gain[0:3].mean(axis=0)
    return rgb_gain


def img_wb_lumi(image):
    image = np.float32(image[..., ::-1]/255.)
    image = image * rgb_gain[None, None]
    image = np.clip(image, 0, 1)
    image = image * illumination_gain
    image = np.clip(image, 0, 1)
    plt.figure()
    plt.imshow(image)
    plt.show()


ideal_linear_rgb = np.float32(np.loadtxt("./data/real_linearRGB_imatest.csv", delimiter=','))
# ideal_lab = np.float32(np.loadtxt("./data/babelcolor_lab_D50.csv", delimiter=','))


#------------analog----------------------------------
analog_gt = np.float32(np.load("./data/analog/analog_gt.npy") / 1000)
# ideal_linear_rgb = smv_colour.XYZ2RGB(torch.from_numpy(np.float32(analog_xyz)), "bt709").numpy()
# print(ideal_linear_rgb)
# print(analog_xyz)
# exit()

ideal_xyz = smv_colour.RGB2XYZ(torch.from_numpy(analog_gt), 'bt709').numpy()
ideal_lab = smv_colour.XYZ2Lab(torch.from_numpy(ideal_xyz)).numpy()
# print(ideal_lab)
# ideal_linear_rgb = smv_colour.XYZ2RGB(torch.from_numpy(np.float32(analog_xyz)), "bt709")
# ideal_lab = analog_xyz / 100.



# for image_path in glob.glob("./data/mindvision/d65_colorchecker.jpg")[0:1]:
for image_path in ["./data/mindvision/d65_colorchecker.jpg"]:
    image = cv2.imread(image_path)
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(image, cv2.mcc.MCC24, 1)
    checkers = detector.getListColorChecker()

    for checker in checkers:
        chartsRGB = checker.getChartsRGB()
        width, height = chartsRGB.shape[:2]
        roi = chartsRGB[0:width, 1]
        rows = int(roi.shape[:1][0])
        src = chartsRGB[:, 1].copy().reshape(int(rows / 3), 1, 3)
        src /= 255.

        src_new = np.squeeze(src, axis=1)[..., ::-1].copy() #  R-G-B channel  shape:(24,3)
        rgb_gain = compute_rgb_gain_from_colorchecker(src_new)
        cc_wb_mean_value = src_new * rgb_gain[None]
        illumination_gain = (ideal_linear_rgb[18:21] / cc_wb_mean_value[18:21]).mean()
        cc_wb_ill_mean_value = illumination_gain * cc_wb_mean_value

        src_for_ccm = cc_wb_ill_mean_value[..., ::-1].copy()  #  B-G-R channel
        src_for_ccm = np.expand_dims(cc_wb_ill_mean_value, 1) #  shape:(24,1,3)
        # src_for_ccm = src_for_ccm ** (1/2.2)

        ideal_lab = np.expand_dims(np.float64(ideal_lab), 1)
        # ideal_linear_rgb = np.expand_dims(np.float64(ideal_linear_rgb), 1)


        #-----------analog---------------------------------------------
        src_for_ccm = np.float32((np.load(r'data/analog/analog_A.npy') + 0) / 255.)
        # src_for_ccm = smv_colour.XYZ2RGB(torch.from_numpy(src_for_ccm), 'bt709').numpy()
        rgb_gain = src_for_ccm[18].max(keepdims=True) / src_for_ccm[18]
        print(rgb_gain)

        src_for_ccm = src_for_ccm * rgb_gain * (0.9 / src_for_ccm[18, :].max())
        print(rgb_gain)
        print(src_for_ccm)
        # print(src_for_ccm.dtype, src_for_ccm.size())
        src_for_ccm = np.float64(src_for_ccm[:, None, :])
        #-------------------------------------------------------------

        model = cv2.ccm_ColorCorrectionModel(src_for_ccm, ideal_lab, cv2.ccm.COLOR_SPACE_Lab_D65_2, np.ones((24,1)))
        # model = cv2.ccm_ColorCorrectionModel(src_for_ccm, ideal_linear_rgb, cv2.ccm.COLOR_SPACE_sRGB)
        # model = cv2.ccm_ColorCorrectionModel(src_for_ccm, cv2.ccm.COLORCHECKER_Macbeth)

        model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        model.setLinearGamma(1)
        model.setLinearDegree(3)
        # model.setSaturatedThreshold(0, 1)
        # model.setSaturatedThreshold(0, 0.98)
        model.run()

        mask = model.getMask()
        for i in range(len(mask)):
            print(i, mask[i])
        print(model.getWeights())

        ccm = model.getCCM()
        # row = [1, 1, 1]
        # ccm = np.vstack([ccm, row])
        print('ccm:\n{}\n'.format(ccm))
        loss = model.getLoss()
        print('loss:\n{}\n'.format(loss))

        print('ccm.sum(axis=0):', ccm.sum(axis=0))
        exit()

        #=====================================================================
        # -- recover img 
        img = np.float32(image[...,::-1].copy() / 255.)
        img_wb = img * rgb_gain[None, None]
        img_wb = np.clip(img_wb, 0, 1)
        img_wb_ill = img_wb * illumination_gain
        img_wb_ill = np.clip(img_wb_ill, 0, 1)
        img_wb_ill = img_wb_ill ** (1/ 2.2)
        cv2.imwrite("./mcc/img_wb_ill.png", img_wb_ill[...,::-1]*255.)


        
        # --- model infer
        calibratedImage = model.infer(img_wb_ill)
        calibratedImage = np.clip(calibratedImage, 0, 1)
        calibratedImage = np.uint8(calibratedImage * 255)
        cv2.imwrite("./mcc/img_wb_ill_ccm.png", calibratedImage)

        # --- verify
        # img_wb_ill_linear = img_wb_ill ** 2.2
        img_wb_ill_linear = gamma_reverse(img_wb_ill)

        img_wb_ill_ccm = np.einsum('ic, hwc->hwi', ccm.T, img_wb_ill_linear)
        

        # img_wb_ill_ccm = img_wb_ill_ccm ** (1/ 2.2)
        img_wb_ill_ccm = gamma(img_wb_ill_ccm)
        
        img_wb_ill_ccm = np.clip(img_wb_ill_ccm, 0, 1)

        img_wb_ill_ccm = np.uint8(img_wb_ill_ccm * 255)
        cv2.imwrite("./mcc/img_wb_ill_ccm_verify.png", img_wb_ill_ccm)

        diff = calibratedImage - img_wb_ill_ccm
        print(diff.max(), diff.mean())
        

        # cv2.imshow('out_img', out_)
        # cv2.imshow("Image", image)
        # cv2.imshow('img_draw', img_draw)
        # # input_img.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # config = {}
        # icc_predict = ImageColorCorrection(config, gt_form='imatest', show_log=False)
        deltaC, deltaE = evaluate_result(img_wb_ill_ccm[..., ::-1]/255., "srgb")
        print('deltaC, deltaE =  ',deltaC.mean(), deltaE.mean())

        deltaC, deltaE = evaluate_result(calibratedImage[..., ::-1]/255., "srgb")
        print('deltaC, deltaE =  ',deltaC.mean(), deltaE.mean())
        