import torch
import torch.nn as nn
from backbone.SegFormer.mix_transformer import mit_b0
from backbone.SegFormer.mix_transformer import mit_b1
from backbone.SegFormer.mix_transformer import mit_b2
from backbone.SegFormer.mix_transformer import mit_b3
from backbone.SegFormer.mix_transformer import mit_b4
from backbone.SegFormer.mix_transformer import mit_b5

class Encoder_RGBT_Segformer(nn.Module):
    def __init__(self, mode):
        super(Encoder_RGBT_Segformer, self).__init__()
        # [16, 32, 64, 128]  Flops 608.01 MMac Params 682.61 k
        if mode == 'b0':
            self.enc_rgb = mit_b0()
            self.enc_t = mit_b0()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b0.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b0.pth")
        if mode == 'b1':
            self.enc_rgb = mit_b1()
            self.enc_t = mit_b1()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b1.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b1.pth")

        if mode == 'b2':
            self.enc_rgb = mit_b2()
            self.enc_t = mit_b2()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b2.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b2.pth")

        if mode == 'b3':
            self.enc_rgb = mit_b3()
            self.enc_t = mit_b3()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b3.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b3.pth")

        if mode == 'b4':
            self.enc_rgb = mit_b4()
            self.enc_t = mit_b4()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b4.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b4.pth")

        if mode == 'b5':
            self.enc_rgb = mit_b5()
            self.enc_t = mit_b5()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b5.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b5.pth")


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        rgb = self.enc_rgb(rgb)
        t = self.enc_t(t)
        return rgb, t


class Encoder_Segformer(nn.Module):
    def __init__(self, mode):
        super(Encoder_Segformer, self).__init__()
        # [16, 32, 64, 128]  Flops 608.01 MMac Params 682.61 k
        if mode == 'b0':
            self.enc_rgb = mit_b0()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b0.pth"
                )
        if mode == 'b1':
            self.enc_rgb = mit_b1()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b1.pth")

        if mode == 'b2':
            self.enc_rgb = mit_b2()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b2.pth")

        if mode == 'b3':
            self.enc_rgb = mit_b3()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b3.pth")


        if mode == 'b4':
            self.enc_rgb = mit_b4()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b4.pth")

        if mode == 'b5':
            self.enc_rgb = mit_b5()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/SegFormer/weight/mit_b5.pth")


    def forward(self, rgb):
        rgb = self.enc_rgb(rgb)
        return rgb
if __name__ == '__main__':
    rgb = torch.randn((1, 3, 512, 640))
    t = torch.randn((1, 3, 512, 640))

    model = Encoder_Segformer(mode='b1')
    out = model(rgb)
    for i in out:
        print(i.shape)


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)