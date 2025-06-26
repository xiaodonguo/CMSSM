import os
import shutil
import json
import time
from torch.cuda import amp
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox.metrics_SUS import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger, AdamW
from toolbox import setup_seed
from Loss.dice import DiceLoss

setup_seed(33)

class eeemodelLoss(nn.Module):

    def __init__(self, device, class_weight=None):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [2.0268,  4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
        self.class_weight_binary = torch.from_numpy(np.array([2.0197, 2.9765])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4584, 18.7187])).float()
        # [ 2.0268,  4.2508, 23.6082, 23.0149, 11.6264, 25.8710] semantic
        # [2.0197, 2.9765] binary
        # [ 1.4584, 18.7187] boundary

        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        self.dice_loss = DiceLoss(mode='multiclass', ignore_index=-1)

    def forward(self, inputs, targets):
        out1 = inputs
        semantic_gt, boundary_gt, binary_gt = targets
        loss1 = self.semantic_loss(out1, semantic_gt)
        # loss2 = self.binary_loss(out2, binary_gt) + self.dice_loss(out2, binary_gt)
        # loss3 = self.boundary_loss(out3, boundary_gt) + self.dice_loss(out3, boundary_gt)
        # loss = loss1 + loss2 + loss3
        return loss1

def run(args):

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    ### multi-gpus
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    ###
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    model = get_model(cfg)
    # model.load_state_dict(torch.load(
    #     '/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/run/b4/200model.pth'))
    print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    ## multi-gpus
    model.to(gpu_ids[0])
    model = torch.nn.DataParallel(model, gpu_ids)

    # dataloader
    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    # val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
    #                           pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True, drop_last=True)
    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()
    train_criterion = eeemodelLoss(gpu_ids[0]).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # # 指标 包含unlabel
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0
    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].cuda()
                label = sample['label'].cuda()
                thermal = sample['thermal'].cuda()
                bound = sample['boundary'].cuda()
                binary = sample['binary'].cuda()
                targets = [label, bound, binary]
            else:
                image = sample['image'].cuda()
                thermal = sample['thermal'].cuda()
                label = sample['label'].cuda()
                bound = sample['boundary'].cuda()
                binary_label = sample['binary'].cuda()
                targets = [label, bound, binary_label]
            with amp.autocast():
                predict = model(image, thermal)[-1]

                loss = train_criterion(predict, targets)

            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            train_loss_meter.update(loss.item())

        scheduler.step()

        # test
        with torch.no_grad():
            model.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].cuda()
                    thermal = sample['thermal'].cuda()
                    label = sample['label'].cuda()
                    predict = model(image)
                else:
                    image = sample['image'].cuda()
                    thermal = sample['thermal'].cuda()
                    label = sample['label'].cuda()
                    predict = model(image, thermal)[-1]


                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [b,c,h,w] to [c, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)


        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]


        # 每轮训练结束后打印结果
        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss:.3f}/{test_loss:.3f}, '
                    f'miou={test_miou:.3f}, ')


        if test_miou > best_test:
            best_test = test_miou
            save_ckpt(logdir, model)

        # if ep >=0.95 * cfg["epochs"]:
        #     name = f"{ep+1}" + "_"
        #     save_ckpt(logdir, model, name)
        #
        # if (ep + 1) % 50 == 0:
        #     name = f"{ep + 1}" + "_"
        #     save_ckpt(logdir, model, name)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/home/ubuntu/code/wild/configs/SUS.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--gpu_ids", type=str, default='0', help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")


    args = parser.parse_args()

    run(args)
