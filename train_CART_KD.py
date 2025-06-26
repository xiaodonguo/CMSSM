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
from Loss.KD_Loss.Logits.KLD import kld_loss
from Loss.KD_loss import kd_ce_loss
from Loss.KD_Loss.Feature.MSE import MSELoss
from Loss.KD_Loss.Feature.CWD import FeatureLoss
from toolbox import setup_seed
from Loss.dice import DiceLoss
setup_seed(33)

class train_Loss(nn.Module):

    def __init__(self):
        super(train_Loss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
                [50.2527, 50.4935, 4.8389, 6.3680, 24.0135, 26.3811, 9.7799, 14.6093, 16.8741, 2.7478, 49.2211, 50.2928])).float()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.dice_loss = DiceLoss(mode='multiclass', ignore_index=-1)
        self.cwd_loss = FeatureLoss(256, 256, 'cwd', 1)

    def forward(self, predict_student, targets, predict_lake, predict_river, predict_coast, predict_ground, f_lake, f_river, f_coast, f_ground, f_s):
        B, C1, H1, W1 = predict_student.size()
        _, C2, H2, W2 = f_s.size()
        semantic_gt = targets
        loss_hard = self.semantic_loss(predict_student, semantic_gt) + self.dice_loss(predict_student, semantic_gt)

        # choose the terrain teacher
        loss_lake = torch.mean(self.cross_entropy(predict_lake, targets), dim=(1, 2)).unsqueeze(1)
        loss_river = torch.mean(self.cross_entropy(predict_river, targets), dim=(1, 2)).unsqueeze(1)
        loss_coast = torch.mean(self.cross_entropy(predict_coast, targets), dim=(1, 2)).unsqueeze(1)
        loss_ground = torch.mean(self.cross_entropy(predict_ground, targets), dim=(1, 2)).unsqueeze(1)
        loss_cat = torch.cat((loss_lake, loss_river, loss_coast, loss_ground), dim=1)
        min_indices = torch.argmin(loss_cat, dim=1)
        teacher_cat = torch.cat((predict_lake.unsqueeze(1), predict_river.unsqueeze(1), predict_coast.unsqueeze(1), predict_ground.unsqueeze(1)), dim=1)
        # teacher_feature_cat = torch.cat((f_lake.unsqueeze(1), f_river.unsqueeze(1), f_coast.unsqueeze(1), f_ground.unsqueeze(1)), dim=1)
        indices1 = min_indices.view(B, 1, 1, 1, 1).expand(-1, -1, C1, H1, W1)
        # indices2 = min_indices.view(B, 1, 1, 1, 1).expand(-1, -1, C2, H2, W2)
        teacher_final = torch.gather(teacher_cat, dim=1, index=indices1).squeeze(1)
        # f_t = torch.gather(teacher_feature_cat, dim=1, index=indices2).squeeze(1)
        # loss_pre = kld_loss(predict_student, teacher_final, 1, False)
        loss_pre = kd_ce_loss(predict_student, teacher_final)
        loss_kd = loss_pre
        # loss_feature = self.cwd_loss([f_s], [f_t])
        # loss_feature = MSELoss(f_s, f_t)
        # loss_kd = (kd_ce_loss(predict_student, predict_lake) + kd_ce_loss(predict_student, predict_coast) + kd_ce_loss(predict_student, predict_river) + kd_ce_loss(predict_student, predict_ground)) / 4

        loss = loss_hard + loss_kd

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
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{"KD"}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    student = get_model(cfg)
    ## multi-gpus
    student.to(gpu_ids[0])
    student = torch.nn.DataParallel(student, gpu_ids)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(student, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)

    teacher_lake = get_model(cfg).to(gpu_ids[0])
    teacher_river =  get_model(cfg).to(gpu_ids[0])
    teacher_coast = get_model(cfg).to(gpu_ids[0])
    teacher_ground = get_model(cfg).to(gpu_ids[0])
    teacher_lake.load_state_dict(torch.load(
        "/run/TCSVT/KD/EfficientVit/2025-05-16-18-59(CART_Terrain-Lake-model1_b1_CM-SSM)/model.pth"))
    teacher_river.load_state_dict(torch.load(
        "/run/TCSVT/KD/EfficientVit/2025-05-16-17-50(CART_Terrain-River-model1_b1_CM-SSM)/model.pth"))
    teacher_coast.load_state_dict(torch.load(
        "/run/TCSVT/KD/EfficientVit/2025-05-16-20-57(CART_Terrain-Coast-model1_b1_CM-SSM)/model.pth"))
    teacher_ground.load_state_dict(torch.load(
        "/run/TCSVT/KD/EfficientVit/2025-05-16-20-58(CART_Terrain-Ground-model1_b1_CM-SSM)/model.pth"))

    # dataloader
    trainset, _, testset1 = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    test_loader1 = DataLoader(testset1, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)
    params_list = student.parameters()
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
        student.train()
        teacher_lake.eval()
        teacher_coast.eval()
        teacher_river.eval()
        teacher_ground.eval()
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
                    predict = student(thermal)
                else:
                    with torch.no_grad():
                        predict_lake, f_lake = teacher_lake(image, thermal)
                        predict_river, f_river = teacher_river(image, thermal)
                        predict_coast, f_coast = teacher_coast(image, thermal)
                        predict_ground, f_ground = teacher_ground(image, thermal)
                    predict_student, f_s = student(image, thermal)
                loss = train_criterion(predict_student, targets, predict_lake, predict_river, predict_coast, predict_ground, f_lake, f_river, f_coast, f_ground, f_s)
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            train_loss_meter.update(loss.item())

        scheduler.step()
        torch.cuda.empty_cache()

        # test
        with torch.no_grad():
            student.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader1):

                image = sample['image'].cuda()
                thermal = sample['thermal'].cuda()
                label = sample['label'].cuda()
                if cfg['inputs'] == 't':
                    predict = student(thermal)
                else:
                    predict = student(image, thermal)[0]

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
                    f'miou={test_miou:.3f}, '
                    )

        if test_miou > best_test:
            best_test = test_miou
            save_ckpt(logdir, student)

        if ep >=0.8 * cfg["epochs"]:
            name = f"{ep+1}" + "_"
            save_ckpt(logdir, student, name)

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
    parser.add_argument("--gpu_ids", type=str, default='0', help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
