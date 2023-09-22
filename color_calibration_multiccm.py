from cProfile import label
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.ImageColorCalibration import ImageColorCorrection
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import smv_colour
import torch
from utils import mcc_detect_color_checker


def calculate_colorchecker_value(image, sorted_centroid, length):
    sorted_centroid2 = np.int32(sorted_centroid)
    mean_value = np.empty((sorted_centroid.shape[0], 3))
    for i in range(len(sorted_centroid)):
        mean_value[i] = np.mean(image[sorted_centroid2[i, 1] - length:sorted_centroid2[i, 1] + length,
                                sorted_centroid2[i, 0] - length:sorted_centroid2[i, 0] + length], axis=(0, 1))
    return np.float32(mean_value)


def generate_calib(image_calib, ccm_space, image_color_space, gt_form, savez_path, ccm_weight=np.ones((24))):
    """This should generate the standalized calibrated CCM.
        'image_calib' is collected under specific illuminant.

    Args:
        image_calib (array): sensor-RGB-IMG
        ccm_space (str): ['srgb','linear']
        image_color_space (str): color-space of the collected image which is used for calibration. ['srgb','linear']
        gt_form (str): decide  ['xrite', 'imatest]
        savez_path (str): save the calibrated CCM and CCT.
    """
    config_minimize = {"method": "minimize", "ccm_space": ccm_space, "gt_form": gt_form, "ccm_weight": ccm_weight}
    icc_minimize = ImageColorCorrection(config_minimize)
    icc_minimize.update_message_from_colorchecker(image_calib)
    image_ccm = icc_minimize.image_correction(image_calib, image_color_space, white_balance=True, illumination_gain=True,
                                                    ccm_correction=True)
    cv2.imwrite('img_ccm.png',image_ccm[...,::-1]**(1/2.2)*255.)
    deltaC, deltaE, _ = icc_minimize.evaluate_result(image_ccm, "linear")

    image_with_gt = icc_minimize.draw_gt_in_image(image_ccm, "linear", deltaE)
    image_with_gt = np.clip(image_with_gt, 0, 1)
    cv2.imwrite('img_ccm_gt.png', image_with_gt[...,::-1]**(1/2.2)*255.)


    print('deltaC, deltaE:  ', deltaC.mean(), deltaE.mean())
    ccm_cur = icc_minimize.ccm
    cct_cur = icc_minimize.cct
    np.savez(savez_path, cct=cct_cur, ccm=ccm_cur)
    return image_ccm, image_with_gt, icc_minimize.sorted_centroid, icc_minimize.cc_mean_value


def generate_weight(cc_rgb_value, min_weight, max_weight, color_index):
    # ideal_lab = np.float32(np.loadtxt("./data/real_lab_imatest.csv", delimiter=','))  # from imatest
    # ideal_xy = smv_colour.XYZ2xyY(smv_colour.Lab2XYZ(torch.from_numpy(ideal_lab))).numpy()[:, 0:2]
    ideal_xy = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(cc_rgb_value), "bt709")).numpy()[:, 0:2]
    c = ideal_xy[color_index]
    distance = np.sum((ideal_xy - c)**2, axis=1) ** 0.5
    print(distance)
    # weight = 1 / (distance + 0.1)
    weight = max_weight - (max_weight-min_weight) * distance / distance.max()

    return weight

# def get_distance(image):

def compute_distance(image, color, sorted_centroid):
    image_ccm = np.float32(image)
    mean_value = calculate_colorchecker_value(image_ccm, sorted_centroid, 50)
    cc_xy = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(mean_value), "bt709")).numpy()[:, 0:2]
    image_ccm_xy = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(image_ccm), "bt709")).numpy()[..., 0:2]
    if color == "r":
        c = cc_xy[14]
    if color == "g":
        c = cc_xy[13]
    if color == "b":
        c = cc_xy[12]
    distance = np.abs(image_ccm_xy - c[None, None]).sum(axis=(2))
    return distance


def compute_distance2(image, cc_mean_value):
    cc_xy = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(cc_mean_value), "bt709")).numpy()[:, 0:2]
    image_ccm_xy = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(image), "bt709")).numpy()[..., 0:2]
    distance = image_ccm_xy[:, :, None, :] - cc_xy[None, None]
    distance = np.abs(distance).sum(axis=(3))
    return distance


if __name__ == '__main__':

    ideal_srgb = np.float32(np.loadtxt("./data/real_linearRGB_imatest.csv", delimiter=',')) ** (1/2.2)

    ccm_space='linear' # srgb or linear
    gt_form='imatest' # imatest or xrite
    savez_path = f"./data/{gt_form}/{ccm_space}/"
    if not os.path.exists(savez_path): os.makedirs(savez_path)

    image_d65 = cv2.imread(r"./data/mindvision/d65_colorchecker.jpg")[..., ::-1] / 255.
    savez_path_d65 = savez_path + "D65_light.npz"

    image_A = cv2.imread(r"./data/mindvision/exposure30.jpg")[..., ::-1] / 255.
    # image_A = cv2.resize(image_A, (image_A.shape[1]//2, image_A.shape[0]//2))[..., ::-1] / 255.
    savez_path_A = savez_path + "A_light.npz"

    image_A = cv2.imread(r"./data/mindvision/mv_2300.PNG")[..., ::-1] / 255.
    savez_path_A = savez_path + "A_light.npz"

    _, image_with_gt, _, cc_mean_value = generate_calib(image_calib=image_A,
                                            ccm_space=ccm_space,
                                            image_color_space='linear',
                                            gt_form=gt_form,
                                            savez_path=savez_path_A,
                                            ccm_weight=np.ones(24))

    # plt.figure()
    # plt.imshow(image_with_gt)
    # plt.show()

    cc_mean_value_xy = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(np.float32(cc_mean_value)), "bt709"))
    # plt.figure()
    # # plt.xlim()
    # for i in range(24):
    #     plt.scatter(cc_mean_value_xy[i, 0], cc_mean_value_xy[i, 1], c=ideal_srgb[i], linewidths=10, label=i+1)
    # plt.legend()
    # plt.show()
    # exit()

    choosen_color = np.array([12, 13, 14])
    result = []
    image_A = np.float32(image_A)
    distance = compute_distance2(image_A, cc_mean_value[choosen_color])
    index = np.argmin(distance, axis=2)
    for i in choosen_color:
        ccm_weight = generate_weight(cc_mean_value, 1, 10, i)
        # ccm_weight[i] =
        # ccm_weight[19:] = 0
        image_ccm, image_with_gt, sorted_centroid, cc_mean_value = generate_calib(image_calib=image_A,
                                                                       ccm_space=ccm_space,
                                                                       image_color_space='linear',
                                                                       gt_form=gt_form,
                                                                       savez_path=savez_path_A,
                                                                       ccm_weight=ccm_weight)
        result.append(image_ccm)

    plt.figure()
    plt.imshow(index)
    result_image = np.empty_like(image_ccm)
    for i in range(image_ccm.shape[0]):
        for j in range(image_ccm.shape[1]):
            result_image[i, j] = result[index[i, j]][i, j]
    config_minimize = {"method": "minimize", "ccm_space": ccm_space, "gt_form": gt_form, "ccm_weight": np.zeros(24)}
    icc_minimize = ImageColorCorrection(config_minimize)
    deltaC, deltaE, _ = icc_minimize.evaluate_result(result_image, "linear")
    image_with_gt = icc_minimize.draw_gt_in_image(result_image, "linear", deltaE)
    print(deltaC[choosen_color].mean(), deltaE[choosen_color].mean())
    print((deltaC.sum() - deltaC[choosen_color].sum())/(24-len(choosen_color)),
          (deltaE.sum() - deltaE[choosen_color].sum())/(24-len(choosen_color)))
    print(deltaC.mean(), deltaE.mean())
    plt.figure()
    plt.imshow(image_with_gt ** (1/2.2))
    plt.show()
    exit()

    # # image_A = cv2.imread(r"./data/tmp/raw_616997531.png")[..., ::-1] / 255.
    # # plt.figure()
    # # plt.imshow(image_A)
    # # plt.show()
    # # savez_path_A = savez_path + "A_light.npz"
    #
    # # ccm_weight = np.array([1, 1, 1, 1, 1, 1,
    # #                        1, 1, 1, 1, 1, 1,
    # #                        1, 1, 1, 1, 1, 1,
    # #                        1, 1, 1, 1, 1, 1])
    # # ccm_weight = np.zeros(24)
    # # ccm_weight[6] = 1
    # # ccm_weight[11] = 1
    # # ccm_weight[14] = 1
    # # ccm_weight[18] = 1
    # # ccm_weight[0] = 1
    #
    # # image_ccm, image_with_gt_r, sorted_centroid = generate_calib(image_calib=image_A,
    # #                                             ccm_space=ccm_space,
    # #                                             image_color_space='linear',
    # #                                             gt_form=gt_form,
    # #                                             savez_path=savez_path_A,
    # #                                             ccm_weight=ccm_weight)
    #
    #
    # # plt.figure()
    # # plt.imshow(image_with_gt_r)
    # # plt.figure()
    # # plt.imshow(image_A)
    # # plt.show()
    #
    # # mean_value = calculate_colorchecker_value(image_d65, sorted_centroid, 50)
    # # ideal_linearrgb = np.float32(np.loadtxt("./data/real_linearRGB_imatest.csv", delimiter=','))  # from imatest
    # # print(ideal_linearrgb.shape)
    # # print(mean_value.shape)
    # # s = np.argsort(ideal_linearrgb[:, 0])
    # # print(s)
    # # plt.figure()
    # # plt.scatter(ideal_linearrgb[:, 0][s], mean_value[:, 0][s])
    # # s = np.argsort(ideal_linearrgb[:, 0])
    # # plt.figure()
    # # plt.scatter(ideal_linearrgb[:, 1][s], mean_value[:, 1][s])
    # # s = np.argsort(ideal_linearrgb[:, 1])
    # # plt.figure()
    # # plt.scatter(ideal_linearrgb[:, 2][s], mean_value[:, 2][s])
    # #
    # # plt.show()
    # # exit()
    #
    # # ccm_weight = np.zeros(24)
    # # ccm_weight[6] = 1
    # # ccm_weight[11] = 1
    # # ccm_weight[14] = 1
    # # ccm_weight[18] = 1
    # # image = np.load("image.npy")
    #
    # # plt.figure()
    # # plt.imshow(g_weight)
    # # plt.figure()
    # # plt.imshow(b_weight)
    # # plt.show()
    # # # distance =
    # # exit()
    #
    #
    #
    # ccm_weight = generate_weight(1, 10, "r")
    # # ccm_weight[14] = 10
    # # ccm_weight = np.array([0, 0, 0, 0, 0, 0,
    # #                        1, 0, 0, 0, 0, 1,
    # #                        0, 0, 1, 0, 1, 0,
    # #                        1, 1, 1, 1, 0, 0])
    # ccm_weight[18:] = 0
    #
    #
    #
    # image_ccm_r, image_with_gt_r, sorted_centroid = generate_calib(image_calib=image_A,
    #                                                                ccm_space=ccm_space,
    #                                                                image_color_space='linear',
    #                                                                gt_form=gt_form,
    #                                                                savez_path=savez_path_A,
    #                                                                ccm_weight=ccm_weight)
    #
    # distance_r = compute_distance(image_ccm_r, 'r', sorted_centroid)
    # ccm_weight = generate_weight(1, 10, "g")
    # ccm_weight[18:] = 0
    #
    # image_ccm_g, image_with_gt_g, sorted_centroid = generate_calib(image_calib=image_A,
    #                                             ccm_space=ccm_space,
    #                                             image_color_space='linear',
    #                                             gt_form=gt_form,
    #                                             savez_path=savez_path_A,
    #                                             ccm_weight=ccm_weight)
    # distance_g = compute_distance(image_ccm_g, 'g', sorted_centroid)
    # ccm_weight = generate_weight(1, 10, "b")
    # ccm_weight[18:] = 0
    # image_ccm_b, image_with_gt_b, sorted_centroid = generate_calib(image_calib=image_A,
    #                                             ccm_space=ccm_space,
    #                                             image_color_space='linear',
    #                                             gt_form=gt_form,
    #                                             savez_path=savez_path_A,
    #                                             ccm_weight=ccm_weight)
    #
    #
    # distance_b = compute_distance(image_ccm_b, 'b', sorted_centroid)
    #
    # r_weight = (distance_r < distance_g) * (distance_r < distance_b)
    # g_weight = (distance_g < distance_r) * (distance_g < distance_b)
    # b_weight = (distance_b < distance_r) * (distance_b < distance_g)
    # new_image = r_weight[..., None] * image_ccm_r + g_weight[..., None] * image_ccm_g + b_weight[..., None] * image_ccm_b
    #
    #
    # config_minimize = {"method": "minimize", "ccm_space": ccm_space, "gt_form": gt_form, "ccm_weight": ccm_weight}
    # icc_minimize = ImageColorCorrection(config_minimize)
    # deltaC, deltaE = icc_minimize.evaluate_result(new_image, "linear")
    # image_with_gt = icc_minimize.draw_gt_in_image(new_image, "linear", deltaE)
    # print(deltaC[0:18].mean(), deltaE[0:18].mean())
    # plt.figure()
    # plt.imshow(image_with_gt)
    #
    #
    # # np.save("./image.npy", new_image)
    # plt.figure()
    # plt.subplot(2, 3, 1)
    # plt.imshow(r_weight)
    # plt.subplot(2, 3, 2)
    # plt.imshow(g_weight)
    # plt.subplot(2, 3, 3)
    # plt.imshow(b_weight)
    # plt.subplot(2, 3, 4)
    # plt.imshow(image_with_gt_r)
    # plt.subplot(2, 3, 5)
    # plt.imshow(image_with_gt_g)
    # plt.subplot(2, 3, 6)
    # plt.imshow(image_with_gt_b)
    # plt.show()
    # # distance = compute_distance(image_ccm, 'r', sorted_centroid)
    # # plt.figure()
    # # plt.imshow(distance)
    # # plt.show()
    # # distance = compute_distance(image_ccm, 'r', sorted_centroid)
    # # plt.figure()
    # # plt.imshow(distance)
    # # plt.show()
    #
    #
    #
    # exit()
