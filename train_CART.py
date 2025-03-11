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
from toolbox.metrics_CART import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger, AdamW
from toolbox import setup_seed
from Loss.dice import DiceLoss
setup_seed(33)

class train_Loss(nn.Module):

    def __init__(self):
        super(train_Loss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
                [50.2527, 50.4935, 4.8389, 6.3680, 24.0135, 26.3811, 9.7799, 14.6093, 16.8741, 2.7478, 49.2211, 50.2928])).float()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.dice_loss = DiceLoss(mode='multiclass', ignore_index=-1)

    def forward(self, inputs, targets):

        out1 = inputs
        semantic_gt = targets
        loss1 = self.semantic_loss(out1, semantic_gt) + self.dice_loss(out1, semantic_gt)
        loss = loss1

        return loss

def run(args):

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    ### multi-gpuss
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    ###
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["terrain"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    model = get_model(cfg)
    # model = create_efficientvit_seg_model(name='efficientvit-seg-b0-cityscapes', weight_url='/root/autodl-tmp/code/backbone/Efficientvit/efficientvit_seg_b0_cityscapes.pt')
    # model.load_state_dict(torch.load(
    #     "/root/autodl-tmp/code/wild/run/2024-10-17-00-10(wild-CMX)/250_model.pth"))
    print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    ## multi-gpus
    model.to(gpu_ids[0])
    model = torch.nn.DataParallel(model, gpu_ids)

    # dataloader
    trainset, valset, _ = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)
    params_list = model.parameters()
    # optimizer = optim.Adam(model.parameters(), lr=cfg['lr_start'])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()
    train_criterion = train_Loss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # # 指标 包含unlabel
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    # running_metrics_test = runningScore(cfg['n_classes'], ignore_index=None)
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
            image = sample['image'].cuda()
            thermal = sample['thermal'].cuda()
            label = sample['label'].cuda()
            targets = label

            with amp.autocast():
                if cfg['inputs'] == 't':
                    predict = model(thermal)
                else:
                    predict = model(image, thermal)
                loss = train_criterion(predict, targets)
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            train_loss_meter.update(loss.item())

        scheduler.step()
        torch.cuda.empty_cache()

        # val
        with torch.no_grad():
            model.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(val_loader):

                image = sample['image'].cuda()
                thermal = sample['thermal'].cuda()
                label = sample['label'].cuda()
                if cfg['inputs'] == 't':
                    predict = model(thermal)
                else:
                    predict = model(image, thermal)

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [b,c,h,w] to [c, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)


        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2

        # 每轮训练结束后打印结果
        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss:.3f}/{test_loss:.3f}, '
                    f'mPA={test_macc:.3f}, '
                    f'miou={test_miou:.3f}, '
                    )
        if test_miou > best_test:
            best_test = test_miou
            save_ckpt(logdir, model)

        # if ep >=0.95 * cfg["epochs"]:
        #     name = f"{ep+1}" + "_"
        #     save_ckpt(logdir, model, name)

        # if (ep + 1) % 50 == 0:
        #     name = f"{ep + 1}" + "_"
        #     save_ckpt(logdir, model, name)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/home/ubuntu/code/wild/configs/CART.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgbt', choices=['rgb', 'rgbt', 't'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--gpu_ids", type=str, default='1', help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
