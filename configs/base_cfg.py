cfg = \
    {
        "model_name": "",

        "inputs": "rgb",                 # rgb, rgbd

        "dataset": "cityscapes",         # nyuv2 | sunrgbd | cityscapes | camvid | irseg
        "root": "../database/{option}",  # nyuv2 | sunrgbd | cityscapes | camvid | irseg
        "n_classes": 20,                 # 41    |    38   |    20      |   12   |   9
        "id_unlabel": 0,                 #   0   |    0    |    0       |   11   |   0
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "p": 0.5,
        "scales_range": "0.5 2.0",
        "crop_size": "736 736",        # 480 640 | 480 640  | 736 736   | 352 480 | 480 640

        "ims_per_gpu": 2,
        "num_workers": 4,

        "lr_start": 5e-4,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_power": 0.9,
        "epochs": 200,

        "loss": "crossentropy",
        "class_weight": "enet"
    }
