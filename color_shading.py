import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np

# import color_correction
import matplotlib.pylab as plt
from utils.ImageColorCalibration import ImageColorCorrection
from tqdm import tqdm
import cv2.cv2 as cv2

image = np.load("image_ccm.npy")
print(image.shape)
image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

delta_C_list = []




for r_shading in tqdm(range(60, 101, 1)):
    temp = []
    for b_shading in range(60, 101, 1):
        # image2 = image.copy()
        image2 = np.copy(image)
        image2[:, :, 0] = image[:, :, 0] * (r_shading / 100.0)
        image2[:, :, 2] = image[:, :, 2] * (b_shading / 100.0)


        config_minimize = {"method": "minimize", "ccm_space": 'linear', "gt_form": 'imatest', "ccm_weight": np.ones((24))}
        icc_minimize = ImageColorCorrection(config_minimize)

        deltaC, deltaE00, deltaE76 = icc_minimize.evaluate_result(image2, "linear")
        image_with_gt = icc_minimize.draw_gt_in_image(image2, "linear", deltaC)
        temp.append(deltaC.mean())
        print(deltaC.mean(), deltaE00.mean())
        cv2.imwrite("./data/shading/%.2f_%.2f.jpg"%((r_shading / 100.0), (b_shading / 100.0)), image_with_gt[:,:,::-1]**(1/2.2)*255.)
    delta_C_list.append(temp)

delta_C_list = np.array(delta_C_list)
np.save("delta_C_list.npy", delta_C_list)
# plt.figure()
# plt.imshow(delta_C_list)
# plt.colorbar()
# plt.show()