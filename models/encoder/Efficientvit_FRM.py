import torch
import torch.nn as nn
from backbone.efficientvit.models.efficientvit.backbone_FRM import efficientvit_backbone_b0
from backbone.efficientvit.models.efficientvit.backbone_FRM import efficientvit_backbone_b1
from backbone.efficientvit.models.efficientvit.backbone_FRM import efficientvit_backbone_b2
from backbone.efficientvit.models.efficientvit.backbone_FRM import efficientvit_backbone_b3



class Encoder_RGBT_Efficientvit(nn.Module):
    def __init__(self, mode):
        super(Encoder_RGBT_Efficientvit, self).__init__()
        # [16, 32, 64, 128]  Flops 608.01 MMac Params 682.61 k
        if mode == 'b0':
            self.enc_rgbt = efficientvit_backbone_b0()
            self.enc_rgbt.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_seg_b0_cityscapes.pt")

        if mode == 'b1':
            self.enc_rgbt = efficientvit_backbone_b1()
            self.enc_rgbt.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_b1_r288.pt")

        if mode == 'b2':
            self.enc_rgbt = efficientvit_backbone_b2()
            self.enc_rgbt.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_b2_r288.pt")

        if mode == 'b3':
            self.enc_rgbt = efficientvit_backbone_b3()
            self.enc_rgbt.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_b3_r288.pt")


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        rgb, t = self.enc_rgbt(rgb, t)
        return rgb, t



if __name__ == '__main__':
    rgb = torch.randn((1, 3, 512, 640))
    t = torch.randn((1, 3, 512, 640))

    model = Encoder_RGBT_Efficientvit(mode='b0')
    out = model(rgb, t)
    for rgb in out[0]:
        print(rgb.shape)
    for t in out[1]:
        print(t.shape)


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)