import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self, rgb_data):
        super(Net, self).__init__()
        self.rgb_data = rgb_data
    def forward(self, ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc4, ccm_calc5, ccm_calc6, ccm_calc7, ccm_calc8, ccm_calc9):
        rgb_tmp = torch.zeros_like(self.rgb_data)
        rgb_tmp[0, :] = (ccm_calc1 * self.rgb_data[0, :] + ccm_calc2 * self.rgb_data[1, :] + ccm_calc3 * self.rgb_data[2, :]) / 1.0
        rgb_tmp[1, :] = (ccm_calc4 * self.rgb_data[0, :] + ccm_calc5 * self.rgb_data[1, :] + ccm_calc6 * self.rgb_data[2, :]) / 1.0
        rgb_tmp[2, :] = (ccm_calc7 * self.rgb_data[0, :] + ccm_calc8 * self.rgb_data[1, :] + ccm_calc9 * self.rgb_data[2, :]) / 1.0
        return rgb_tmp

class Net(nn.Module):
    def __init__(self, ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc4, ccm_calc5, ccm_calc6, ccm_calc7, ccm_calc8, ccm_calc9):
        super(Net, self).__init__()
        self.ccm_calc1=ccm_calc1
        self.ccm_calc2=ccm_calc2
        self.ccm_calc3=ccm_calc3
        self.ccm_calc4=ccm_calc4
        self.ccm_calc5=ccm_calc5
        self.ccm_calc6=ccm_calc6
        self.ccm_calc7=ccm_calc7
        self.ccm_calc8=ccm_calc8
        self.ccm_calc9=ccm_calc9
    def forward(self, rgb_data):
        rgb_tmp = torch.zeros_like(rgb_data)
        rgb_tmp[0, :] = (self.ccm_calc1 * rgb_data[0, :] + self.ccm_calc2 * rgb_data[1, :] + self.ccm_calc3 * rgb_data[2, :]) / 1.0
        rgb_tmp[1, :] = (self.ccm_calc4 * rgb_data[0, :] + self.ccm_calc5 * rgb_data[1, :] + self.ccm_calc6 * rgb_data[2, :]) / 1.0
        rgb_tmp[2, :] = (self.ccm_calc7 * rgb_data[0, :] + self.ccm_calc8 * rgb_data[1, :] + self.ccm_calc9 * rgb_data[2, :]) / 1.0
        return rgb_tmp