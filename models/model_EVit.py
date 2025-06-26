import torch
import torch.nn as nn
from proposed.encoder.Efficientvit import Encoder_Efficientvit
from proposed.encoder.Efficientvit import Encoder_RGBT_Efficientvit
from proposed.decoder.MLP import Decoder_MLP
from proposed.decoder.MLP_plus import Decoder_MLP_plus
import torch.nn.functional as F
# from proposed.fuison_strategy.base_fusion import Fusion_Module
from proposed.fuison_strategy.fusion import Fusion_Module
from proposed.decoder.DeepLabV3 import DeepLabHeadV3Plus

class Model(nn.Module):
    def __init__(self, mode, inputs, n_class=9, fusion_mode='max', share_weights=False):
        super(Model, self).__init__()
        if mode == 'b0':
            channels = [16, 32, 64, 128]
            emb_c = 128
        elif mode == 'b1':
            channels = [32, 64, 128, 256]
            emb_c = 128
        elif mode == 'b2':
            channels = [48, 96, 192, 384]
            emb_c = 256
        elif mode in ['b3', 'l1', 'l2']:
            channels = [64, 128, 256, 512]
            emb_c = 768
        elif mode == 'l3':
            channels = [128, 256, 512, 1024]
            emb_c = 1024
        self.inputs = inputs
        if inputs == 'unimodal':
            self.encoder = Encoder_Efficientvit(mode=mode)
        if inputs == 'rgbt':
            self.encoder = Encoder_RGBT_Efficientvit(mode=mode, share_weights=share_weights)
            self.fusion_module = Fusion_Module(fusion_mode=fusion_mode, channels=channels)
        self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = DeepLabHeadV3Plus(in_channels=channels[-1], low_level_channels=channels[0], num_classes=12)

    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        if self.inputs == 'unimodal':
            fusions = self.encoder(rgb)
        else:
            f_rgb, f_t = self.encoder(rgb, t)
            fusions = self.fusion_module(f_rgb, f_t)
        sem, _c = self.decoder(fusions)
        sem = F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)
        return sem


if __name__ == '__main__':
    device = torch.device("cuda")
    rgb = torch.rand(1, 3, 480, 640).to(device)
    t = torch.rand(1, 3, 480, 640).to(device)
    model = Model(mode='b1', inputs='rgbt', fusion_mode='CM-SSM', n_class=12).eval().to(device)
    out = model(rgb, t)
    print(out.shape)

    from ptflops import get_model_complexity_info

    # flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    # print('Flops ' + flops)
    # print('Params ' + params)

    from fvcore.nn import flop_count_table, FlopCountAnalysis

    print(flop_count_table(FlopCountAnalysis(model, rgb)))


    # from thop import profile
    # flops, params = profile(model, inputs=(rgb, t))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")