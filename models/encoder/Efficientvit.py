import torch
import torch.nn as nn
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_b0
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_b1
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_b2
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_b3
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_l0
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_l1
from backbone.efficientvit.models.efficientvit import efficientvit_backbone_l2

class Encoder_RGBT_Efficientvit(nn.Module):
    def __init__(self, mode):
        super(Encoder_RGBT_Efficientvit, self).__init__()
        # [16, 32, 64, 128]  Flops 608.01 MMac Params 682.61 k
        if mode == 'b0':
            self.enc_rgb = efficientvit_backbone_b0()
            self.enc_t = efficientvit_backbone_b0()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_seg_b0_cityscapes.pt")
            self.enc_t.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_seg_b0_cityscapes.pt")
        if mode == 'b1':
            self.enc_rgb = efficientvit_backbone_b1()
            self.enc_t = efficientvit_backbone_b1()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b1_r288.pt")
            self.enc_t.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b1_r288.pt")

        if mode == 'b2':
            self.enc_rgb = efficientvit_backbone_b2()
            self.enc_t = efficientvit_backbone_b2()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b2_r288.pt")
            self.enc_t.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b2_r288.pt")

        if mode == 'b3':
            self.enc_rgb = efficientvit_backbone_b3()
            self.enc_t = efficientvit_backbone_b3()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b3_r288.pt")
            self.enc_t.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b3_r288.pt")

        if mode == 'l1':
            self.enc_rgb = efficientvit_backbone_l1()
            self.enc_t = efficientvit_backbone_l1()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_l1_r224.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_l1_r224.pt")

        if mode == 'l2':
            self.enc_rgb = efficientvit_backbone_l2()
            self.enc_t = efficientvit_backbone_l2()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_l2_r384.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_l2_r384.pt")


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        rgb = self.enc_rgb(rgb)
        t = self.enc_t(t)
        # t = self.enc_rgb(t)
        return rgb, t


class Encoder_Efficientvit(nn.Module):
    def __init__(self, mode):
        super(Encoder_Efficientvit, self).__init__()
        # [16, 32, 64, 128]  Flops 608.01 MMac Params 682.61 k
        if mode == 'b0':
            self.enc_rgb = efficientvit_backbone_b0()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_seg_b0_cityscapes.pt"
                )
        if mode == 'b1':
            self.enc_rgb = efficientvit_backbone_b1()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b1_r288.pt")

        if mode == 'b2':
            self.enc_rgb = efficientvit_backbone_b2()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b2_r288.pt")

        if mode == 'b3':
            self.enc_rgb = efficientvit_backbone_b3()
            self.enc_rgb.init_weights(
                pretrained="/home/ubuntu/code/backbone/efficientvit/efficientvit_b3_r288.pt")


        if mode == 'l1':
            self.enc_rgb = efficientvit_backbone_l1()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_l1_r224.pt")

        if mode == 'l2':
            self.enc_rgb = efficientvit_backbone_l2()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_l2_r384.pt")


    def forward(self, rgb):
        rgb = self.enc_rgb(rgb)
        return rgb
if __name__ == '__main__':
    rgb = torch.randn((1, 3, 600, 960))
    t = torch.randn((1, 3, 600, 960))

    model = Encoder_Efficientvit(mode='b1')
    out = model(rgb)
    for i in out:
        print(i.shape)


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)