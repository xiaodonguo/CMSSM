import torch
import torch.nn as nn
from proposed.encoder.ConvNeXtV2 import Encoder_ConvNeXt
from proposed.encoder.ConvNeXtV2 import Encoder_RGBT_ConvNeXt
from proposed.decoder.MLP import Decoder_MLP
import torch.nn.functional as F
from proposed.fuison_strategy.fusion import Fusion_Module


class Model(nn.Module):
    def __init__(self, mode, inputs, n_class=9, fusion_mode='max', share_weights=False):
        super(Model, self).__init__()
        self.inputs = inputs
        assert mode in ['atto', 'femto', 'pico', 'nano', 'tiny', 'base']
        self.mode = mode
        if mode == 'atto':
            channels = [40, 80, 160, 320]
            emb_c = 128
        elif mode == 'femto':
            channels = [48, 96, 192, 384]
            emb_c = 256
        elif mode == 'pico':
            channels = [64, 128, 256, 512]
            emb_c = 768
        elif mode == 'nano':
            channels = [80, 160, 320, 640]
            emb_c = 768
        elif mode == 'tiny':
            channels = [96, 192, 384, 768]
            emb_c = 768
        elif mode == 'base':
            channels = [128, 256, 512, 1024]
            emb_c = 768
        if inputs == 'single':
            self.encoder = Encoder_ConvNeXt(mode=mode)
        else:
            self.encoder = Encoder_RGBT_ConvNeXt(mode=mode, share_weights=share_weights)
            self.fusion_module = Fusion_Module(fusion_mode=fusion_mode, channels=channels)
        self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class)


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        if self.inputs == 'single':
            fusions = self.encoder(rgb)
        else:
            f_rgb, f_t = self.encoder(rgb, t)
            fusions = self.fusion_module(f_rgb, f_t)
            # if self.mode == 'atto':
            #     for i in range(4):
            #         fusions[i] = self.connector[i](fusions[i])
        sem, _ = self.decoder(fusions)
        sem = F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)
        # bina = F.interpolate(bina, scale_factor=4, mode='bilinear', align_corners=False)
        # bound = F.interpolate(bound, scale_factor=4, mode='bilinear', align_corners=False)
        return sem


if __name__ == '__main__':
    rgb = torch.rand(1, 3, 512, 640).cuda()
    t = torch.rand(1, 3, 512, 640).cuda()
    model = Model(mode='atto', inputs='rgbt', n_class=12, fusion_mode='CM-SSM', share_weights=False).eval().cuda()
    out = model(rgb, t)
    # print(out.shape)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)

    # from thop import profile
    # flops, params = profile(model, inputs=(rgb, t))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")
