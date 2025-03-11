import torch
import torch.nn.functional as F
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernelsize, stride=1, padding=0, dialation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, padding=padding, stride=stride,
                              dilation=dialation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DSC(nn.Module):
    def __init__(self, inchannels, outchannels, kenelsize, padding, dilation):
        super(DSC, self).__init__()
        self.depthwiseConv = nn.Conv2d(inchannels, inchannels, kenelsize, groups=inchannels, padding=padding, dilation=dilation)
        self.pointwiseConv = nn.Conv2d(inchannels, outchannels, 1)
        self.BN = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.depthwiseConv(x)
        x = self.pointwiseConv(x)
        x = self.relu(self.BN(x))
        return x

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, t):
        max_out, _ = torch.max(torch.cat((rgb, t), dim=1), dim=1, keepdim=True)
        out1 = self.conv1(max_out)
        weight_rgb = self.sigmoid(out1)
        weight_t = 1 - weight_rgb
        rgb_out = rgb * weight_rgb
        t_out = t * weight_t
        return rgb_out, t_out

class CAM(nn.Module):
    def __init__(self, inplanes, outplanes, ratio):
        super(CAM, self).__init__()
        self.inplanes = inplanes // 2
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.FC1 = nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.FC2 = nn.Conv2d(inplanes // ratio, outplanes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, t):
        out = self.FC2(self.relu1(self.FC1(self.maxpool(torch.cat((rgb, t), dim=1)))))
        channel_weight = self.sigmoid(out)
        out1 = torch.mul(torch.cat((rgb, t), dim=1), channel_weight)
        rgb = out1[:, 0:self.inplanes, :, :]
        t = out1[:, self.inplanes:2*self.inplanes, :, :]
        return rgb, t

class MHFI_change(nn.Module):
    def __init__(self, inplanes, former_inplanes=None, NO=None):
        super(MHFI_change, self).__init__()
        self.NO = NO
        self.SA = SAM()
        self.CA = CAM(2 * inplanes, 2 * inplanes, 16)
        self.dilation1 = Dilation(inplanes)
        self.dilation2 = Dilation(inplanes)
        self.interaction1 = Interaction(inplanes // 4)
        self.interaction2 = Interaction(inplanes // 4)
        self.interaction3 = Interaction(inplanes // 4)
        self.interaction4 = Interaction(inplanes // 4)
        self.conv1 = BasicConv2d(inplanes, inplanes, 1, 1, 0)
        if NO != 1:
            self.conv2 = BasicConv2d(former_inplanes, inplanes, 1, 2, 0)
        self.conv3_1 = BasicConv2d(inplanes, inplanes, 1, 1, 0)
        self.conv3_2 = BasicConv2d(2 * inplanes, inplanes, 1, 1, 0)
        # self.pool = F.interpolate(scale_factor=0.5, mode='bilinear', align_corners=False)
    def forward(self, rgb, t, former=None):
        H, W = rgb.shape[-2], rgb.shape[-1]
        if self.NO == 1 or self.NO == 2:
            rgb, t = self.SA(rgb, t)
        else:
            rgb, t = self.CA(rgb, t)
        p1 = rgb + t
        p2 = torch.mul(rgb, t)
        add1, add2, add3, add4 = self.dilation1(p1)  # in_c // 4
        mul1, mul2, mul3, mul4 = self.dilation2(p2)  # in_c // 4
        interaction1 = self.interaction1(add1, mul1, former=None, NO=1)  # in_c // 4
        interaction2 = self.interaction2(add2, mul2, former=interaction1, NO=2)
        interaction3 = self.interaction3(add3, mul3, former=interaction2, NO=3)
        interaction4 = self.interaction4(add4, mul4, former=interaction3, NO=4)
        dilation_out = torch.cat((interaction1, interaction2, interaction3, interaction4), dim=1) # in_c
        dilation_out = self.conv1(dilation_out)
        out = p1 + dilation_out + p2
        if self.NO == 1:
            out = self.conv3_1(out)
        else:
            former = self.conv2(former)
            out = self.conv3_2(torch.cat((former, out), dim=1))
        return out

class Interaction(nn.Module):
    def __init__(self, in_c):
        super(Interaction, self).__init__()
        self.conv3_1 = BasicConv2d(in_c * 2, in_c, 3, 1, 1)
        self.conv3_2 = BasicConv2d(in_c * 3, in_c, 3, 1, 1)
    def forward(self, add, mul, former, NO):
        if NO == 1:
            out = torch.cat((add, mul), dim=1)
            out = self.conv3_1(out)
        else:
            out = torch.cat((add, mul, former), dim=1)
            out = self.conv3_2(out)
        return out

class Dilation(nn.Module):
    def __init__(self, in_c):
        super(Dilation, self).__init__()
        self.dilation1 = DSC(in_c, in_c // 4, 3, 3, 3)
        self.dilation2 = DSC(in_c, in_c // 4, 3, 5, 5)
        self.dilation3 = DSC(in_c, in_c // 4, 3, 7, 7)
        self.dilation4 = DSC(in_c, in_c // 4, 3, 9, 9)
    def forward(self, input):
        out1 = self.dilation1(input)
        out2 = self.dilation2(input)
        out3 = self.dilation3(input)
        out4 = self.dilation4(input)
        return out1, out2, out3, out4

if __name__ == '__main__':
    former = torch.randn((2, 64, 128, 160))
    rgb = torch.randn((2, 128, 64, 80))
    t = torch.randn((2, 128, 64, 80))
    model = Fusion(128, 64, 1)
    out = model(rgb, t, former)
    print(out.shape)