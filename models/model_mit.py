import torch
import torch.nn as nn
from proposed.encoder.Segformer import Encoder_Segformer
from proposed.encoder.Segformer import Encoder_RGBT_Segformer
from proposed.decoder.MLP import Decoder_MLP
import torch.nn.functional as F
from proposed.fuison_strategy.fusion import Fusion_Module
from proposed.decoder.DeepLabV3 import DeepLabHeadV3Plus

class Model(nn.Module):
    def __init__(self, mode, inputs, n_class=12, num_heads=[1, 2, 4, 8], norm_fuse=nn.BatchNorm2d, fusion_mode='max'):
        super(Model, self).__init__()
        if mode == 'b0':
            channels = [32, 64, 160, 256]
        else:
            channels = [64, 128, 320, 512]
        if mode in ['b0', 'b1']:
            emb_c = 128
        else:
            emb_c = 768
        self.inputs = inputs
        if inputs == 'single':
            self.encoder = Encoder_Segformer(mode=mode)
        if inputs == 'rgbt':
            self.encoder = Encoder_RGBT_Segformer(mode=mode)
            self.fusion_module = Fusion_Module(fusion_mode=fusion_mode, channels=channels, num_heads=num_heads,
                                               norm_fuse=norm_fuse)
        self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = DeepLabHeadV3Plus(in_channels=channels[-1], low_level_channels=channels[0], num_classes=12)

    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        if self.inputs == 'single':
            fusions = self.encoder(rgb)
        else:
            f_rgb, f_t = self.encoder(rgb, t)
            fusions = self.fusion_module(f_rgb, f_t)
        sem, _ = self.decoder(fusions)
        sem = F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)
        return sem


if __name__ == '__main__':
    rgb = torch.rand(1, 3, 512, 640).cuda()
    t = torch.rand(1, 3, 512, 640).cuda()
    model = Model(mode='b0', n_class=12, fusion_mode='CM-SSM-KD', inputs='rgbt').eval().cuda()
    out = model(rgb, t)
    print(out.shape)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 512, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)

    # from thop import profile
    # flops, params = profile(model, inputs=(rgb, t))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")

    # Flops 325.02 GMac
    # Params 239.62 M