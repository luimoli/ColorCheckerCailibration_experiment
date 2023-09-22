import numpy as np
import colour
import cv2.cv2 as cv2
from utils import smv_colour
import torch

ideal_lab_d50 = torch.from_numpy(np.float32(np.loadtxt("./data/real_lab_d50_3ns.csv", delimiter=','))) 

ideal_xyz_d50 = smv_colour.Lab2XYZ(ideal_lab_d50, 'd50')

ideal_rgb = smv_colour.XYZ2RGB(ideal_xyz_d50, 'bt709')

ideal_srgb = ideal_rgb ** (1/2.2) * 255.

print(ideal_srgb)
