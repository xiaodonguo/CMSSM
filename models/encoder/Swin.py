import torch
import torch.nn as nn
from backbone.Swin_transformer.swin_transformer import SwinTransformer_t
from backbone.Swin_transformer.swin_transformer import SwinTransformer_s
from backbone.Swin_transformer.swin_transformer import SwinTransformer_b


class Encoder_RGBT_Swin(nn.Module):
    def __init__(self, mode):
        super(Encoder_RGBT_Swin, self).__init__()
        # [96, 196, 384, 768] Flops 60.02 GMac Params 55.04 M
        if mode == 'tiny':
            self.enc_rgb = SwinTransformer_t()
            self.enc_t = SwinTransformer_t()
            # self.enc_rgb.init_weights(pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_tiny_patch4_window7_224.pth")
            self.enc_rgb.init_weights(pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_tiny_patch4_window7_224_22k.pth")

            # self.enc_t.init_weights(pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_tiny_patch4_window7_224.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_tiny_patch4_window7_224_22k.pth")

        # [96, 196, 384, 768] Flops 117.14 GMac Params 97.68 M
        if mode == 'small':
            self.enc_rgb = SwinTransformer_s()
            self.enc_t = SwinTransformer_s()
            # self.enc_rgb.init_weights(
            #     pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_small_patch4_window7_224.pth")
            # self.enc_t.init_weights(
            #     pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_small_patch4_window7_224.pth")
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_small_patch4_window7_224_22k.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_small_patch4_window7_224_22k.pth")
        # [128, 256, 512, 1024] Flops 208.07 GMac Params 173.49 M
        if mode == 'base':
            self.enc_rgb = SwinTransformer_b()
            self.enc_t = SwinTransformer_b()
            # self.enc_rgb.init_weights(
            #     pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_base_patch4_window7_224.pth")
            # self.enc_t.init_weights(
            #     pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_base_patch4_window7_224.pth")
            self.enc_rgb.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_base_patch4_window7_224_22k.pth")
            self.enc_t.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_base_patch4_window7_224_22k.pth")


    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        rgb = self.enc_rgb(rgb)
        t = self.enc_t(t)
        return rgb, t


class Encoder_Swin(nn.Module):
    def __init__(self, mode):
        super(Encoder_Swin, self).__init__()
        # [96, 196, 384, 768] Flops 60.02 GMac Params 55.04 M
        if mode == 'tiny':
            self.enc = SwinTransformer_t()
            # self.enc.init_weights(pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_tiny_patch4_window7_224.pth")
            self.enc.init_weights(pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_tiny_patch4_window7_224_22k.pth")


        # [96, 196, 384, 768] Flops 117.14 GMac Params 97.68 M
        if mode == 'small':
            self.enc_rgb = SwinTransformer_s()
            # self.enc.init_weights(
            #     pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_small_patch4_window7_224.pth")
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_small_patch4_window7_224_22k.pth")
        # [128, 256, 512, 1024] Flops 208.07 GMac Params 173.49 M
        if mode == 'base':
            self.enc = SwinTransformer_b()
            # self.enc_rgb.init_weights(
            #     pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_base_patch4_window7_224.pth")
            self.enc.init_weights(
                pretrained="/root/autodl-tmp/code/backbone/Swin_transformer/swin_base_patch4_window7_224_22k.pth")


    def forward(self, input):
        features = self.enc(input)
        return features


if __name__ == '__main__':
    rgb = torch.randn((1, 3, 512, 640))
    t = torch.randn((1, 3, 512, 640))

    model = Encoder_RGBT_Swin(mode='tiny')
    out = model(rgb, t)
    for i in out[0]:
        print(i.shape)


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)