
import os
import time
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from toolbox.metrics_SUS import averageMeter, runningScore
from toolbox import class_to_RGB
from toolbox.datasets.SUS import SUS
from toolbox import get_model
# from proposed.teacher.teacher import Model
# from proposed.student.student import Model

def evaluate(logdir, save_predict=False, options=['val', 'test', 'test_day', 'test_night'], prefix=''):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda:0')

    loaders = []
    for opt in options:
        dataset = SUS(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
        cmap = dataset.cmap

    model = get_model(cfg).to(device)
    # model = Model(name='base', num_classes=6).to(device)
    model.load_state_dict(torch.load(args.model_weight, map_location=device), strict=True)
    # running_metrics_val = runningScore(cfg['n_classes'], ignore_index=0)
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()
    save_path = os.path.join('./result/SUS', '1')
    if not os.path.exists(save_path) and save_predict:
        os.makedirs(save_path)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
                time_start = time.time()
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)['sem']

                else:
                    image = sample['image'].to(device)
                    thermal = sample['thermal'].to(device)
                    label = sample['label'].to(device)
                    # predict = model(image, thermal)['sem']
                    predict = model(image, thermal)[-1]
                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w] 按照第一个维度求最大值，并返回最大值对应的索引
                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)
                time_meter.update(time.time() - time_start, n=image.size(0))

                if save_predict:
                    predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                    predict = class_to_RGB(predict, N=len(cmap), cmap=cmap)  # 如果数据集没有给定cmap,使用默认cmap
                    predict = Image.fromarray(predict)
                    predict.save(os.path.join(save_path, sample['label_path'][0]))

            

        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        for k, v in metrics[0].items():
            print(k, f'{v * 100:.1f}')

        print('iou for each class .....')
        for k, v in metrics[1].items():
            print(k, f'{v * 100:.1f}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, default="/home/ubuntu/code/wild/run/2025-06-11-22-46(SUS-MDNet)")
    parser.add_argument("--model_weight", type=str,
                        default="/home/ubuntu/code/wild/run/2025-06-11-22-46(SUS-MDNet)/model.pth")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")
    args = parser.parse_args()
    evaluate(args.logdir, save_predict=args.s, options=['test'], prefix='')
