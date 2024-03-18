import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#----------------------------------------------------------------------------------------------------------------------
# Local Feature Fusion Module
class LFFM(nn.Module):
    def __init__(self, channels):
        super(LFFM, self).__init__()
        self.Conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.Conv2 = nn.Conv2d(2 * channels, channels, 3, stride=1, padding=2, bias=True)
        self.Conv3 = nn.Conv2d(channels, channels // 2, 3, stride=1, padding=0, dilation=1)
        self.Conv4 = nn.Conv2d(channels, channels // 2, 3, stride=1, padding=2, dilation=3)
        self.Conv5 = nn.Conv2d(channels, channels // 2, 3, stride=1, padding=1, bias=True)
        self.Conv6 = nn.Conv2d(channels // 2, 1, 1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(channels // 2)
        self.RELU = nn.ReLU()
    def forward(self, x):
        out1 = self.Conv1(x)
        out2 = self.Conv2(torch.cat([out1, x], 1))
        out3_1 = self.RELU(self.bn(self.Conv3(out2)))
        out3_2 = self.RELU(self.bn(self.Conv4(out2)))
        out4 = torch.cat([out3_1, out3_2], 1)
        out5 = self.RELU(self.bn(self.Conv5(out4)))
        out6 = self.Conv6(out5)
        return out6
#-----------------------------------------------------------------------------------------------------------------------
#channel attention block
class CAB(nn.Module):
    def __init__(self):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return x * out.expand_as(x)
#-----------------------------------------------------------------------------------------------------------------------a
# Multi-Scale Adaptive Feature Fusion Module
class MSRB_Block(nn.Module):
    """
    Considering the computing power requirements,
    I did not use Concat in the open source code to connect the features.
    Here I am using ADD. your device if you can, you can use 'torch.cat' method
    """
    def __init__(self, channel):
        super(MSRB_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)
        self.CAB = CAB()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        output1 = self.relu(self.conv1(x))
        output2 = self.relu(self.conv2(x))
        output3 = self.relu(self.conv3(x))
        output4 = self.relu(self.conv4(x))
        out1 = output1 + output2 + output3 + output4
        output5 = self.relu(self.conv1(out1))
        output6 = self.relu(self.conv2(out1))
        output7 = self.relu(self.conv3(out1))
        output8 = self.relu(self.conv4(out1))
        out2 = output5 + output6 + output7 + output8
        out3 = torch.add(x, out2)
        output = self.CAB(out3)
        output = self.conv5(output)
        return output
#-----------------------------------------------------------------------------------------------------------------------
class MMFF_NET(nn.Module):
    def __init__(self):
        super(MMFF_NET, self).__init__()
        self.conv_x = nn.Conv2d(1, 64, 3, padding=1)
        self.conv0_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv1_1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool4 = nn.AvgPool2d(2, stride=1, ceil_mode=True)
        self.pool5 = nn.AvgPool2d(2, stride=1, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3_1_down = nn.Conv2d(128, 64, 1)
        self.conv3_2_down = nn.Conv2d(128, 64, 1)
        self.conv4_1_down = nn.Conv2d(256, 64, 1)
        self.conv4_2_down = nn.Conv2d(256, 64, 1)
        self.conv5_1_down = nn.Conv2d(256, 64, 1)
        self.conv5_2_down = nn.Conv2d(256, 64, 1)
        self.conv5_3_down = nn.Conv2d(256, 64, 1)
        self.score_dsn0 = nn.Conv2d(64, 1, 1)
        self.score_dsn1 = nn.Conv2d(64, 1, 1)
        self.score_dsn2 = nn.Conv2d(64, 1, 1)
        self.score_dsn3 = nn.Conv2d(64, 1, 1)
        self.score_dsn4 = nn.Conv2d(64, 1, 1)
        self.score_dsn5 = nn.Conv2d(64, 1, 1)
        self.score_fuse = nn.Conv2d(1, 1, 1)
        self.residual0_1 = MSRB_Block(channel=16)
        self.residual0_2 = MSRB_Block(channel=16)
        self.residual1_1 = MSRB_Block(channel=32)
        self.residual1_2 = MSRB_Block(channel=32)
        self.residual2_1 = MSRB_Block(channel=64)
        self.residual2_2 = MSRB_Block(channel=64)
        self.residual3_1 = MSRB_Block(channel=64)
        self.residual3_2 = MSRB_Block(channel=64)
        self.residual4_1 = MSRB_Block(channel=64)
        self.residual4_2 = MSRB_Block(channel=64)
        self.residual5_1 = MSRB_Block(channel=64)
        self.residual5_2 = MSRB_Block(channel=64)
        self.residual5_3 = MSRB_Block(channel=64)
        self.weight_deconv1 = self._make_bilinear_weights(2, 1).cuda()
        self.weight_deconv2 = self._make_bilinear_weights(4, 1).cuda()
        self.weight_deconv3 = self._make_bilinear_weights(8, 1).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(24, 1).cuda()
        self.W_Change = nn.Conv2d(128, 64, 1)
        self.W_Change_1 = nn.Conv2d(192, 64, 1)
        self.bottle = nn.Conv2d(in_channels=1 * 6, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.LFFM = LFFM(8)
#------------------------------------------------------------------------------------------------------------------------
    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w
#-----------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        conv0_1 = self.relu(self.conv0_1(x))
        conv0_2 = self.relu(self.conv0_2(conv0_1))
        pool1 = self.pool1(conv0_2)
        conv1_1 = self.relu(self.conv1_1(pool1))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool2 = self.pool2(conv1_2)
        conv2_1 = self.relu(self.conv2_1(pool2))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool3 = self.pool3(conv2_2)
        conv3_1 = self.relu(self.conv3_1(pool3))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        pool4 = self.pool4(conv3_2)
        conv4_1 = self.relu(self.conv4_1(pool4))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        pool5 = self.pool5(conv4_2)
        conv5_1 = self.relu(self.conv5_1(pool5))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))
        conv0_1_down = self.residual0_1(conv0_1)
        conv0_2_down = self.residual0_2(conv0_2)
        conv1_1_down = self.residual1_1(conv1_1)
        conv1_2_down = self.residual1_2(conv1_2)
        conv2_1_down = self.residual2_1(conv2_1)
        conv2_2_down = self.residual2_2(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_1_down = self.residual3_1(conv3_1_down)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_2_down = self.residual3_2(conv3_2_down)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_1_down = self.residual4_1(conv4_1_down)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_2_down = self.residual4_2(conv4_2_down)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_1_down = self.residual5_1(conv5_1_down)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_2_down = self.residual5_2(conv5_2_down)
        conv5_3_down = self.conv5_3_down(conv5_3)
        conv5_3_down = self.residual5_3(conv5_3_down)
        Cann0 = torch.cat([conv0_1_down, conv0_2_down], 1)
        Cann0_out = self.W_Change(Cann0)
        Cann1 = torch.cat([conv1_1_down, conv1_2_down], 1)
        Cann1_out = self.W_Change(Cann1)
        Cann2 = torch.cat([conv2_1_down, conv2_2_down], 1)
        Cann2_out = self.W_Change(Cann2)
        Cann3 = torch.cat([conv3_1_down, conv3_2_down], 1)
        Cann3_out = self.W_Change(Cann3)
        Cann4 = torch.cat([conv4_1_down, conv4_2_down], 1)
        Cann4_out = self.W_Change(Cann4)
        Cann5 = torch.cat([conv5_1_down, conv5_2_down, conv5_3_down], 1)
        Cann5_out = self.W_Change_1(Cann5)
        out0 = self.score_dsn0(Cann0_out)
        out1 = self.score_dsn1(Cann1_out)
        out2 = self.score_dsn2(Cann2_out)
        out3 = self.score_dsn3(Cann3_out)
        out4 = self.score_dsn4(Cann4_out)
        out5 = self.score_dsn5(Cann5_out)
        out1 = F.conv_transpose2d(out1, self.weight_deconv1, stride=2)
        out2 = F.conv_transpose2d(out2, self.weight_deconv2, stride=4)
        out3 = F.conv_transpose2d(out3, self.weight_deconv3, stride=8)
        out4 = F.conv_transpose2d(out4, self.weight_deconv4, stride=8)
        out5 = F.conv_transpose2d(out5, self.weight_deconv5, stride=8)
        fuse_enhance_img = torch.cat([out0, out1, out2, out3, out4, out5], 1)
        fuse_enhance_img1 = self.bottle(fuse_enhance_img)
        y = x + out0*(torch.pow(x, 2) - x)
        y = y + out1*(torch.pow(y, 2) - y)
        y = y + out2*(torch.pow(y, 2) - y)
        y = y + out3*(torch.pow(y, 2) - y)
        y = y + out4*(torch.pow(y, 2) - y)
        y = y + out5*(torch.pow(y, 2) - y)
        enhance_img = y + fuse_enhance_img1 * (torch.pow(y, 2) - y)
        HR = torch.cat([x, enhance_img, out0, out1, out2, out3, out4, out5], 1)
        Fu = self.LFFM(HR)
        r =torch.cat([out0, out1, out2, out3, out4, out5, fuse_enhance_img1], 1)
        return enhance_img, r, Fu
