import torch
import torch.nn as nn
from backbone.covnextV2.convnextv2 import convnextv2_atto
from backbone.covnextV2.convnextv2 import convnextv2_femto
from backbone.covnextV2.convnextv2 import convnextv2_pico
from backbone.covnextV2.convnextv2 import convnextv2_nano
from backbone.covnextV2.convnextv2 import convnextv2_tiny
from backbone.covnextV2.convnextv2 import convnextv2_base


class Encoder_ConvNeXt(nn.Module):
    def __init__(self, mode):
        super(Encoder_ConvNeXt, self).__init__()
        # [40, 80, 160, 320] Flops 7.19 GMac Params 6.77 M
        if mode == 'atto':
            self.enc = convnextv2_atto()
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_atto_1k_224_ema.pt")
        # [48, 96, 192, 384] Flops 9.7 GMac Params 10.23 M
        if mode == 'femto':
            self.enc = convnextv2_femto()
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_femto_1k_224_ema.pt")
        # [64, 128, 256, 512] Flops 17.92 GMac Params 17.11 M
        if mode == 'pico':
            self.enc= convnextv2_pico()
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_pico_1k_224_ema.pt")
        # [80, 160, 320, 640] Flops 32.03 GMac Params 29.97 M
        if mode == 'nano':
            self.enc = convnextv2_nano()
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_nano_1k_224_ema.pt")
        # [96, 192, 384, 768] Flops 58.33 GMac Params 55.73 M
        if mode == 'tiny':
            self.enc = convnextv2_tiny()
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_tiny_1k_224_ema.pt")
        # [128, 256, 512, 1024] Flops 200.85 GMac Params 175.39 M
        if mode == 'base':
            self.enc = convnextv2_base()
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_base_1k_224_ema.pt")


    def forward(self, rgb):
        rgb = self.enc(rgb)
        return rgb

class Encoder_RGBT_ConvNeXt(nn.Module):
    def __init__(self, mode):
        super(Encoder_RGBT_ConvNeXt, self).__init__()
        # [40, 80, 160, 320] Flops 7.19 GMac Params 6.77 M
        if mode == 'atto':
            self.enc_rgb = convnextv2_atto()
            self.enc_t = convnextv2_atto()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_atto_1k_224_ema.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_atto_1k_224_ema.pt")
        # [48, 96, 192, 384] Flops 9.7 GMac Params 10.23 M
        if mode == 'femto':
            self.enc_rgb = convnextv2_femto()
            self.enc_t = convnextv2_femto()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_femto_1k_224_ema.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_femto_1k_224_ema.pt")
        # [64, 128, 256, 512] Flops 17.92 GMac Params 17.11 M
        if mode == 'pico':
            self.enc_rgb = convnextv2_pico()
            self.enc_t = convnextv2_pico()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_pico_1k_224_ema.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_pico_1k_224_ema.pt")
        # [80, 160, 320, 640] Flops 32.03 GMac Params 29.97 M
        if mode == 'nano':
            self.enc_rgb = convnextv2_nano()
            self.enc_t = convnextv2_nano()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_nano_22k_384_ema.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_nano_22k_384_ema.pt")
        # [96, 192, 384, 768] Flops 58.33 GMac Params 55.73 M
        if mode == 'tiny':
            self.enc_rgb = convnextv2_tiny()
            self.enc_t = convnextv2_tiny()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_tiny_22k_384_ema.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_tiny_22k_384_ema.pt")
        # [128, 256, 512, 1024] Flops 200.85 GMac Params 175.39 M
        if mode == 'base':
            self.enc_rgb = convnextv2_base()
            self.enc_t = convnextv2_base()
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_base_22k_384_ema.pt")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/ConvNextV2/convnextv2_base_22k_384_ema.pt")


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        rgb = self.enc_rgb(rgb)
        t = self.enc_t(t)
        return rgb, t

if __name__ == '__main__':
    rgb = torch.randn((1, 3, 512, 640))
    t = torch.randn((1, 3, 512, 640))

    model = Encoder_ConvNeXt(mode='base')
    out = model(rgb, t)
    for i in out[0]:
        print(i.shape)


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)