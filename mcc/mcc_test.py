import glob
# import cv2.cv2 as cv2
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
# from utils.CCM_function2 import ImageColorCorrection
# from utils.minimize_ccm import gamma, gamma_reverse

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

# for image_path in glob.glob("./data/mindvision/d65_colorchecker.jpg")[0:1]:
ideal_linear_rgb = np.float32(np.loadtxt("./data/real_linearRGB_imatest.csv", delimiter=','))
ideal_lab = np.float32(np.loadtxt("./data/real_lab_imatest.csv", delimiter=','))

for image_path in ["./data/mindvision/d65_colorchecker.jpg"]:
    image = cv2.imread(image_path)
    # image = image / 255
    # image = image ** (1 / 2.2) *255
    # image = np.uint8(image)

    # plt.figure()
    # plt.imshow(image[..., ::-1].copy())
    # plt.show()

    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(image, cv2.mcc.MCC24, 1)

    checkers = detector.getListColorChecker()
    for checker in checkers:
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        img_draw = image.copy()
        cdraw.draw(img_draw)
        chartsRGB = checker.getChartsRGB()
        width, height = chartsRGB.shape[:2]
        roi = chartsRGB[0:width, 1]
        rows = int(roi.shape[:1][0])
        src = chartsRGB[:, 1].copy().reshape(int(rows / 3), 1, 3)
        # print(src)
        src /= 255

        src_new = np.squeeze(src, axis=1)[..., ::-1].copy() # R-G-B channel  shape:(24,3)
        rgb_gain = compute_rgb_gain_from_colorchecker(src_new)
        cc_wb_mean_value = src_new * rgb_gain[None]
        illumination_gain = (ideal_linear_rgb[18:21] / cc_wb_mean_value[18:21]).mean()
        cc_wb_ill_mean_value = illumination_gain * cc_wb_mean_value

        src_for_ccm = cc_wb_ill_mean_value[..., ::-1].copy()  #  B-G-R channel
        src_for_ccm = np.expand_dims(cc_wb_ill_mean_value, 1) #  shape:(24,1,3)
        # src_for_ccm = src_for_ccm ** (1/2.2)


        # model = cv2.ccm_ColorCorrectionModel(src_for_ccm, cv2.ccm.COLORCHECKER_Macbeth)
        ideal_lab = np.expand_dims(np.float64(ideal_lab), 1)
        ideal_linear_rgb = np.expand_dims(np.float64(ideal_linear_rgb), 1)
        # model = cv2.ccm_ColorCorrectionModel(src_for_ccm, ideal_lab, cv2.ccm.COLOR_SPACE_Lab_D65_2)
        model = cv2.ccm_ColorCorrectionModel(src_for_ccm, ideal_linear_rgb, cv2.ccm.COLOR_SPACE_sRGB)

        # model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)

        model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        model.setLinearGamma(2.2)
        model.setLinearDegree(3)

        model.setSaturatedThreshold(0, 1)
        # model.setSaturatedThreshold(0, 0.98)

        model.run()
        print(model.getWeights())

        ccm = model.getCCM()
        row = [1, 1, 1]
        ccm = np.vstack([ccm, row])

        print('ccm:\n{}\n'.format(ccm))
        loss = model.getLoss()
        print('loss:\n{}\n'.format(loss))

        ###############################################################################################
        #  input_img = Image.open('download.jpeg', 'r').convert("RGB")
        #  input_img = deGamma(input_img, gamma=2.2)
        #  input_img = sRGB2XYZ(input_img)
        #  input_img = correctColor(input_img, ccm)
        #  input_img = XYZ2sRGB(input_img)
        #  out_img = applyGamma(input_img, gamma=2.2)
        #  input_img.save('output.jpg');
        ###############################################################################################
        # img = cv2.imread('with.jpeg')
        img_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_ = img_.astype(np.float64)
        img_ = img_ / 255
        # img_ = img_[..., ::-1].copy()
        # image = np.float32(image[..., ::-1]/255.)
        img_ = img_ * rgb_gain[None, None]
        img_ = np.clip(img_, 0, 1)
        img_ = img_ * illumination_gain
        img_ = np.clip(img_, 0, 1)
        img_ = img_ ** (1/ 2.2)
        cv2.imwrite("img_.png", img_[...,::-1]*255.)

        calibratedImage = model.infer(img_)
        out_ = calibratedImage * 255
        out_[out_ < 0] = 0
        out_[out_ > 255] = 255
        out_ = out_.astype(np.uint8)
        # out_ = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test.png", out_)
        # cv2.imshow('out_img', out_)
        # cv2.imshow("Image", image)
        # cv2.imshow('img_draw', img_draw)
        # # input_img.show()
        # cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()
        # config = {}
        # icc_predict = ImageColorCorrection(config, gt_form='imatest', show_log=False)
        # deltaC, deltaE = icc_predict.evaluate_result(out_[..., ::-1]/255., "srgb")
        # print('deltaC, deltaE =  ',deltaC.mean(), deltaE.mean())
        print(ccm.sum(axis=1))