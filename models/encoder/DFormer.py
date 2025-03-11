import torch
import torch.nn as nn
from backbone.Dformer.DFormer import DFormer_Tiny
from backbone.Dformer.DFormer import DFormer_Small
from backbone.Dformer.DFormer import DFormer_Base
from backbone.Dformer.DFormer import DFormer_Large


class Encoder_RGBT_DFormer(nn.Module):
    def __init__(self, mode):
        super(Encoder_RGBT_DFormer, self).__init__()
        # [16, 32, 64, 128]  Flops 608.01 MMac Params 682.61 k
        if mode == 'tiny':
            self.enc = DFormer_Tiny()
            self.enc.init_weights(
                pretrained="/home/ubuntu/code/backbone/DFormer/RGBD/DFormer_Tiny.pth.tar"
                # pretrained="/home/ubuntu/code/backbone/DFormer/RGBT/tiny.pth.tar"
            )
        if mode == 'small':
            self.enc = DFormer_Small()
            # self.enc.init_weights("/home/ubuntu/code/backbone/DFormer/RGBD/DFormer_Small.pth.tar")
            self.enc.init_weights("/home/ubuntu/code/backbone/DFormer/RGBT/Small.pth.tar")
        if mode == 'base':
            self.enc = DFormer_Base()
            self.enc.init_weights(
                pretrained="/home/ubuntu/code/backbone/DFormer/RGBD/DFormer_Base.pth.tar")

        if mode == 'large':
            self.enc = DFormer_Large()
            self.enc.init_weights(
                pretrained="/home/ubuntu/code/backbone/DFormer/RGBD/DFormer_Large.pth.tar")


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        outs = self.enc(rgb, t)
        return outs

if __name__ == '__main__':
    rgb = torch.randn((1, 3, 512, 640)).cuda()
    t = torch.randn((1, 3, 512, 640)).cuda()

    model = Encoder_RGBT_DFormer(mode='large').eval().cuda()
    out = model(rgb)
    for i in out:
        print(i.shape)


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)