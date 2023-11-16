import torch
import torch.nn as nn
import torch.nn.functional as F
from untils import gradient, MS_SSIM
#-----------------------------------------------------------------------------------------------------------------------
def L_spa(org, enhance):
    kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
    kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
    kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
    kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
    weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
    weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
    weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
    weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
    pool = nn.AvgPool2d(4)
    org_mean = torch.mean(org, 1, keepdim=True)
    enhance_mean = torch.mean(enhance, 1, keepdim=True)
    org_pool = pool(org_mean)
    enhance_pool = pool(enhance_mean)
    D_org_letf = F.conv2d(org_pool, weight_left, padding=1)
    D_org_right = F.conv2d(org_pool, weight_right, padding=1)
    D_org_up = F.conv2d(org_pool, weight_up, padding=1)
    D_org_down = F.conv2d(org_pool, weight_down, padding=1)
    D_enhance_letf = F.conv2d(enhance_pool, weight_left, padding=1)
    D_enhance_right = F.conv2d(enhance_pool, weight_right, padding=1)
    D_enhance_up = F.conv2d(enhance_pool, weight_up, padding=1)
    D_enhance_down = F.conv2d(enhance_pool, weight_down, padding=1)
    D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
    D_right = torch.pow(D_org_right - D_enhance_right, 2)
    D_up = torch.pow(D_org_up - D_enhance_up, 2)
    D_down = torch.pow(D_org_down - D_enhance_down, 2)
    E = (D_left + D_right + D_up + D_down)
    return E
#-----------------------------------------------------------------------------------------------------------------------
def L_exp(x):
    mean_val = 0.6
    pool = nn.AvgPool2d(16)
    x = torch.mean(x,1,keepdim=True)
    mean = pool(x)
    d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
    return d
#-----------------------------------------------------------------------------------------------------------------------
def L_TV(x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        TV_return = 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return TV_return
def HR_LOSS(HR_img, LR_img):
    MS_SSIM_1 = MS_SSIM(max_val=1)
    G_HR = gradient(HR_img)
    G_LR = gradient(LR_img)
    MSSSIM_HLR = MS_SSIM_1.ms_ssim(G_HR, G_LR)
    img_aver_positive = 1-MSSSIM_HLR
    return img_aver_positive



