import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Block, self).__init__()
        self.BN = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU()

    def forward(self, input1, input2, input3, input4):
        input = torch.cat((input1, input2, input3, input4), dim=1)
        out = self.BN(input)
        out = self.conv1(out)
        out = self.BN1(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.ReLU(out)
        return out

class FSN_Decoeder(nn.Module):
    def __init__(self, channels=[64, 128, 320, 512]):
        super(FSN_Decoeder, self).__init__()
        self.channels = channels
        self.decoder1 = Block(4 * channels[3], 2 * channels[3], channels[3])
        self.decoder2 = Block(4 * channels[2], 2 * channels[2], channels[2])
        self.decoder3 = Block(4 * channels[1], 2 * channels[1], channels[1])
        self.decoder4 = Block(4 * channels[0], 2 * channels[0], channels[0])
        # stage1
        self.tp1_1 = nn.Sequential(nn.Conv2d(channels[0], channels[3], 1, 1, 0),
                                   nn.Upsample(scale_factor=1/8, mode="bilinear", align_corners=True))
        self.tp1_2 = nn.Sequential(nn.Conv2d(channels[1], channels[3], 1, 1, 0),
                                   nn.Upsample(scale_factor=1/4, mode="bilinear", align_corners=True))
        self.tp1_3 = nn.Sequential(nn.Conv2d(channels[2], channels[3], 1, 1, 0),
                                   nn.Upsample(scale_factor=1/2, mode="bilinear", align_corners=True))
        # stage2
        self.tp2_1 = nn.Sequential(nn.Conv2d(channels[0], channels[2], 1, 1, 0),
                                   nn.Upsample(scale_factor=1/4, mode="bilinear", align_corners=True))
        self.tp2_2 = nn.Sequential(nn.Conv2d(channels[1], channels[2], 1, 1, 0),
                                   nn.Upsample(scale_factor=1/2, mode="bilinear", align_corners=True))
        self.tp2_4 = nn.Sequential(nn.Conv2d(channels[3], channels[2], 1, 1, 0),
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))

        # stage3
        self.tp3_1 = nn.Sequential(nn.Conv2d(channels[0], channels[1], 1, 1, 0),
                                   nn.Upsample(scale_factor=1/2, mode="bilinear", align_corners=True))
        self.tp3_3 = nn.Sequential(nn.Conv2d(channels[2], channels[1], 1, 1, 0),
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        self.tp3_4 = nn.Sequential(nn.Conv2d(channels[3], channels[1], 1, 1, 0),
                                   nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True))

        # stage4
        self.tp4_2 = nn.Sequential(nn.Conv2d(channels[1], channels[0], 1, 1, 0),
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        self.tp4_3 = nn.Sequential(nn.Conv2d(channels[2], channels[0], 1, 1, 0),
                                   nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True))
        self.tp4_4 = nn.Sequential(nn.Conv2d(channels[3], channels[0], 1, 1, 0),
                                   nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True))

        self.final = nn.Sequential(nn.Conv2d(channels[0], 9, 3, 1, 1),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
    def forward(self, input1, input2, input3, input4):
        # decoder1
        in_11 = self.tp1_1(input1)
        in_12 = self.tp1_2(input2)
        in_13 = self.tp1_3(input3)
        out1 = self.decoder1(in_11, in_12, in_13, input4)

        # decoder2
        in_21 = self.tp2_1(input1)
        in_22 = self.tp2_2(input2)
        in_24 = self.tp2_4(out1)
        out2 = self.decoder2(in_21, in_22, input3, in_24)

        # decoder3
        in_31 = self.tp3_1(input1)
        in_33 = self.tp3_3(out2)
        in_34 = self.tp3_4(out1)
        out3 = self.decoder3(in_31, input2, in_33, in_34)

        # decoder4
        in_42 = self.tp4_2(out3)
        in_43 = self.tp4_3(out2)
        in_44 = self.tp4_4(out1)
        out4 = self.decoder4(input1, in_42, in_43, in_44)

        out = self.final(out4)

        return out

if __name__ == '__main__':
    input = [torch.randn(2, 64, 120, 160),
             torch.randn(2, 128, 60, 80),
             torch.randn(2, 320, 30, 40),
             torch.randn(2, 512, 15, 20)]

    Decoder = FSN_Decoeder(channels=[64, 128, 320, 512])
    print("==> Total params: %.2fM" % (sum(p.numel() for p in Decoder.parameters()) / 1e6))
    out = Decoder(input[0], input[1], input[2], input[3])
    print(out.shape)




