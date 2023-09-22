import matplotlib.pyplot as plt
import torch
import numpy as np
import smv_colour
import torch.nn as nn

from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
from deltaE.deltaE import DeltaE_cmp
from plot_func import plot_two_subfig

# ccm = torch.tensor([[1655, -442, -189], [-248, 1466, -194], [-48, -770, 1842]], dtype=torch.float32)
# rgb_data = torch.randint(0, 255, (3, 100))
# rgb_data = rgb_data.float()

delta = DeltaE_cmp()
rgb_data = torch.from_numpy(np.float32(genfromtxt("./data/measure_rgb_ck2.csv", delimiter=','))).permute(1,0)
rgb_ideal = torch.from_numpy(np.float32(genfromtxt("./data/real_rgb.csv", delimiter=',')) ** 2.2).permute(1,0)


# error_manual = torch.randn((3, 100)) * 16
# rgb_target = ccm.mm(rgb_data)/1.0
# rgb_target_error = rgb_target + error_manual

ccm_calc1 = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
ccm_calc2 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc3 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc4 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc5 = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
ccm_calc6 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc7 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc8 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc9 = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)


def deltaE76(Lab1, Lab2):
    d_E = torch.linalg.norm(Lab1 - Lab2, axis=-1)
    return d_E

def squared_loss(rgb_tmp, rgb_ideal):
    rgb_ideal_ = rgb_ideal.permute(1,0)
    realxyz = smv_colour.RGB2XYZ(rgb_ideal_, 'bt709')
    reallab = smv_colour.XYZ2Lab(realxyz)
    rgb_tmp_ = rgb_tmp.permute(1,0)
    cxyz = smv_colour.RGB2XYZ(rgb_tmp_, 'bt709')
    clab = smv_colour.XYZ2Lab(cxyz)
    # deltaE = deltaE76(clab, reallab)
    deltaE = delta.delta_E_CIE2000(clab, reallab)

    # return torch.mean(torch.sqrt(deltaE))
    return torch.mean(deltaE)

    # return torch.sum((rgb_tmp-rgb_ideal)**2)

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size

def net(ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc4, ccm_calc5, ccm_calc6, ccm_calc7, ccm_calc8, ccm_calc9, rgb_data):
    rgb_tmp = torch.zeros_like(rgb_data)
    rgb_tmp[0, :] = (ccm_calc1 * rgb_data[0, :] + ccm_calc2 * rgb_data[1, :] + ccm_calc3 * rgb_data[2, :]) / 1.0
    rgb_tmp[1, :] = (ccm_calc4 * rgb_data[0, :] + ccm_calc5 * rgb_data[1, :] + ccm_calc6 * rgb_data[2, :]) / 1.0
    rgb_tmp[2, :] = (ccm_calc7 * rgb_data[0, :] + ccm_calc8 * rgb_data[1, :] + ccm_calc9 * rgb_data[2, :]) / 1.0
    return rgb_tmp

lr = 0.03
num_epochs = 1000

for epoch in range(num_epochs):
    # if epoch >= 2500:
    #     lr = 0.01
    # # if epoch >= 5000:
    # #     lr = 0.01
    l = squared_loss(net(ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc4, ccm_calc5, ccm_calc6, ccm_calc7, ccm_calc8, ccm_calc9, rgb_data), rgb_ideal)
    l.backward()
    sgd([ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc4, ccm_calc5, ccm_calc6, ccm_calc7, ccm_calc8, ccm_calc9], lr, 96)
    ccm_calc1.grad.data.zero_()
    ccm_calc2.grad.data.zero_()
    ccm_calc3.grad.data.zero_()
    ccm_calc5.grad.data.zero_()
    ccm_calc6.grad.data.zero_()
    ccm_calc7.grad.data.zero_()
    print('epoch %d, loss %f'%(epoch, l))

ccm = torch.tensor([[ccm_calc1, ccm_calc2, ccm_calc3],
                    [ccm_calc4, ccm_calc5, ccm_calc6],
                    [ccm_calc7, ccm_calc8, ccm_calc9]], dtype=torch.float32)
print(ccm)

rgb_apply_ccm = ccm.mm(rgb_data)/1.0   # [3,3] * [3, num_of_points] = [3, num_of_points]
# rgb_apply_ccm = rgb_data.mm(ccm) / 1.0

plot_two_subfig(rgb_data, rgb_ideal, rgb_apply_ccm)
