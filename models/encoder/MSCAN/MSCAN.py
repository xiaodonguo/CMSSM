import torch
import torch.nn as nn
from backbone.MSCAN.mscan import MSCAN_tiny
from backbone.MSCAN.mscan import MSCAN_base
from backbone.MSCAN.mscan import MSCAN_small
from backbone.MSCAN.mscan import MSCAN_large

class Encoder_MSCAN(nn.Module):
    def __init__(self, mode):
        super(Encoder_MSCAN, self).__init__()
        # [32, 64, 120, 160] Flops 11.17 GMac Params 7.81 M
        if mode == 'tiny':
            self.enc_rgb = MSCAN_tiny()
            self.enc_t = MSCAN_tiny()
        # [64, 128, 320, 512] Flops 31.78 GMac Params 26.9 M
        if mode == 'small':
            self.enc_rgb = MSCAN_small()
            self.enc_t = MSCAN_small()
        # [64, 128, 320, 512] Flops 62.99 GMac Params 52.56 M
        if mode == 'base':
            self.enc_rgb = MSCAN_base()
            self.enc_t = MSCAN_base()
        # [64, 128, 320, 512] Flops 111.97 GMac Params 89.27 M
        if mode == 'large':
            self.enc_rgb = MSCAN_large()
            self.enc_t = MSCAN_large()
        self.enc_rgb.init_weights()
        self.enc_t.init_weights()

    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        rgb = self.enc_rgb(rgb)
        t = self.enc_t(t)
        return rgb, t

if __name__ == '__main__':
    rgb = torch.rand(8, 3, 480, 640)
    t = torch.rand(8, 3, 480, 640)
    encoder = Encoder_MSCAN(mode='large')
    feature = encoder(rgb, t)
    for i in feature:
        for e in i:
            print(e.shape)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(encoder, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in encoder.parameters()) / 1e6))