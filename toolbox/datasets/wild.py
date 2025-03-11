import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 这里的0是GPU设备的索引，你可以根据需要进行更改


class Wild(data.Dataset):

    def __init__(self, cfg, mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre_feature_all-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug
        self.rgb_infos = []
        self.t_infos = []
        self.label_infos = []

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
            self.binary_class_weight = np.array([1.5121, 10.2388])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, 'rgbt_splits', f'rgb_{mode}.txt'), 'r') as f:
            for line in f:
                paths = line.strip().split(',')
                self.rgb_infos.append(paths[0].strip())
                self.label_infos.append(paths[1].strip())

        with open(os.path.join(self.root, 'rgbt_splits', f'thermal16_{mode}.txt'), 'r') as f:
            for line in f:
                paths = line.strip().split(',')
                self.t_infos.append(paths[0].strip())

    def __len__(self):
        return len(self.rgb_infos)

    def __getitem__(self, index):
        rgb_path = self.rgb_infos[index].strip()
        thermal_path = self.t_infos[index].strip()
        label_path = self.label_infos[index].strip()

        image = Image.open(os.path.join(self.root, rgb_path))
        # thermal = Image.open(os.path.join(self.root, thermal_path)).convert('RGB')  # 读取8位图像
        # 读取16位图像并进行预处理
        thermal = cv2.imread(os.path.join(self.root, thermal_path), cv2.IMREAD_UNCHANGED)
        thermal = preprocess_image(thermal).convert('RGB')
        label = Image.open(os.path.join(self.root, label_path))

        sample = {
            'image': image,
            'thermal': thermal,
            'label': label,
        }

        if self.mode in ['train'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['thermal'] = self.dp_to_tensor(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [(255, 36, 0),
                (0, 0, 0),
                (242, 216, 196),
                (89, 70, 54),
                (166, 166, 166),
                (82, 89, 90),
                (155, 230, 0),
                (0, 138, 53),
                (0, 216, 245),
                (13, 127, 252),
                (255, 249, 0),
                (254, 0, 170)]

def preprocess_image(image):
    # 归一化到 0-1 之间
    p1, p99 = np.percentile(image, (1, 99))
    image_rescaled = np.clip((image - p1) / (p99 - p1), 0, 1)
    normalized_data = image_rescaled.astype(np.float32)
    # 转换为0-255的uint8类型
    image_uint8 = (normalized_data * 255).astype(np.uint8)

    # 应用CLAHE的时候必须得先转为uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_uint8)

    return Image.fromarray(image_clahe)


if __name__ == '__main__':
    import json

    path = "/configs/CART.json"
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    dataset = Wild(cfg, mode='test', do_aug=True)
    print(len(dataset))
    # train/val/test
    # 1731/272/279
    from toolbox.utils import ClassWeight

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
                                               num_workers=cfg['num_workers'], pin_memory=True)
    classweight = ClassWeight('enet')  # enet, median_freq_balancing
    class_weight = classweight.get_weight(train_loader, 12)
    class_weight = torch.from_numpy(class_weight).float()

    print(class_weight)
    # weight
    # tensor([50.2527, 50.4935, 4.8389, 6.3680, 24.0135, 26.3811, 9.7799, 14.6093, 16.8741, 2.7478, 49.2211, 50.2928])