import sys
sys.path.append('../')
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import torch
import torch
import torch.nn as nn
import argparse
import os
import time
from torchvision import models, transforms
from torch.utils.data import DataLoader
import shutil
import json
import collections
import datetime
import my_optim
import torch.nn.functional as F
from models import *
from torch.autograd import Variable
from utils import AverageMeter
from utils import Metrics
from utils.LoadData import data_loader, data_loader2
from utils.Restore import restore
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
ROOT_DIR = '/data1/mengmeng/STNET/deit_taformer'
IMG_DIR = '/data1/Dataset/CUB_200_2011/images'
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshot_bins')

train_list = os.path.join(ROOT_DIR, 'datalist', 'CUB', 'train.txt')
test_list = os.path.join(ROOT_DIR, 'datalist', 'CUB', 'test.txt')

LR = 0.0001  # 0.0002
EPOCH = 50  # 50
DISP_INTERVAL = 20

def get_arguments():
    parser = argparse.ArgumentParser(description='vit')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                        help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default='',
                        help='Directory of training images')
    parser.add_argument("--train_list", type=str,
                        default=train_list)
    parser.add_argument("--test_list", type=str,
                        default=test_list)
    parser.add_argument("--batch_size", type=int, default=20)  # 20
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--arch", type=str, default='vgg_v0')  # vgg_v1
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disp_interval", type=int, default=20)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='False')
    parser.add_argument("--restore_from", type=str, default='/data1/mengmeng/STNET/deit_taformer/pretrained/deit_small_patch16_224-cd65a155.pth')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = eval(args.arch).model(pretrained=False,
                                  num_classes=args.num_classes,
                                  args=args)  # pretrained=False
    model.cuda()
    model = torch.nn.DataParallel(model, range(args.num_gpu))

    optimizer = my_optim.get_finetune_optimizer(args, model)
    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return model, optimizer

class GradualWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_diversity = AverageMeter()
    loss_self = AverageMeter()
    loss_tri = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model, optimizer = get_model(args)
    model.train()
    train_loader, _ = data_loader(args)

    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('#epoch,loss,pred@1,pred@5\n')


    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    end = time.time()
    max_iter = total_epoch*len(train_loader)
    print('Max iter:', max_iter)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch + 1, eta_min=1e-5)  #1e-4
    scheduler = GradualWarmupScheduler(optimizer, multiplier=5, total_epoch=5, after_scheduler=cos_scheduler)
    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        loss_diversity.reset()
        loss_self.reset()
        loss_tri.reset()
        top1.reset()
        top5.reset()
        batch_time.reset()
        scheduler.step(current_epoch)

        for idx, dat in enumerate(train_loader):
            img_path, img, label, img_idx = dat
            global_counter += 1
            img, label = img.cuda(),  label.cuda()
            img_var, label_var = Variable(img), Variable(label)
            label_p = 201*torch.ones_like(label_var)
            logits, out_y, mask = model(img_var)
            loss_x = model.module.get_loss(logits, label_var)
            loss_y = model.module.get_loss(out_y, label_p)
            mask_gt = torch.ones(mask.size(0), mask.size(1), mask.size(2)*mask.size(3)).cuda()
            mu = torch.ones(2).cuda()
            nu = torch.ones(mask.size(2)*mask.size(3)).cuda()
            for i in range(len(mask_gt[0])):
                cost,pi = SK(mu,nu, 1-mask[i].reshape(2, mask.size(2)*mask.size(3)))
                # Rescale pi so that the max pi for each gt equals to 1.
                rescale_factor, _ = pi.max(dim=1)
                pi = pi / rescale_factor.unsqueeze(1)
                mask_gt[i] = pi

            ## hard_label
            np_f = mask_gt[:,0,:].data.cpu().numpy()
            np_g = mask_gt[:,1,:].data.cpu().numpy()
            gt_f = np.where(np_f>np_g, 1, 0)
            gt_g = np.where(np_f<=np_g, 1, 0)
            gt_ft = torch.from_numpy(gt_f).float().cuda()
            gt_gt = torch.from_numpy(gt_g).float().cuda()
            mask_gt_hard = torch.cat([gt_ft.unsqueeze(1), gt_gt.unsqueeze(1)], dim=1)
            loss_sk = model.module.loss_bce(mask.reshape(mask.size(0), mask.size(1), -1), mask_gt_hard)
            # if current_epoch <= 5:
            #     loss = loss_x + 0.005*loss_y
            # else:
            loss = loss_x + 0.01*loss_y + 0.05*loss_sk.mean()
            self_loss = loss_y.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if not args.onehot == 'True':
                logits1 = torch.squeeze(logits)
                prec1_1, prec5_1 = Metrics.accuracy(logits1.data, label.long(), topk=(1, 5))
                top1.update(prec1_1[0], img.size()[0])
                top5.update(prec5_1[0], img.size()[0])

            losses.update(loss_x.data, img.size()[0])
            loss_diversity.update(loss_sk.mean().data, img.size()[0])
            loss_self.update(self_loss.data, img.size()[0])
            # loss_tri.update(triplet_loss.data, img.size()[0])
            batch_time.update(time.time() - end)

            end = time.time()
            if global_counter % 1000 == 0:
                losses.reset()
                loss_diversity.reset()
                loss_self.reset()
                loss_tri.reset()
                top1.reset()
                top5.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss_x {loss_self.val:.4f} ({loss_self.avg:.4f})\t'
                      'Loss_y {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_sk {diversity.val:.4f} ({diversity.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader), batch_time=batch_time, loss_self=loss_self,
                    loss = losses, diversity=loss_diversity, top1=top1, top5=top5))

        if current_epoch >= 48:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'arch': 'resnet',
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d_glo_step_%d.pth.tar'
                                     %(args.dataset, current_epoch, global_counter))

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            fw.write('%d,%.4f,%.3f,%.3f\n'%(current_epoch, losses.avg, top1.avg, top5.avg))

        current_epoch += 1


def SK(mu, nu, C):
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    # Sinkhorn iterations
    for i in range(10):
        v = 0.1 * \
            (torch.log(
                nu + 1e-8) - torch.logsumexp(Mu(C, u, v).transpose(-2, -1), dim=-1)) + v
        u = 0.1 * \
            (torch.log(
                mu + 1e-8) - torch.logsumexp(Mu(C, u, v), dim=-1)) + u

    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    pi = torch.exp(
        Mu(C, U, V)).detach()
    # Sinkhorn distance
    cost = torch.sum(
        pi * C, dim=(-2, -1))
    return cost, pi

def Mu(C, u, v):
    '''
    "Modified cost for logarithmic updates"
    "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
    '''
    return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / 0.1


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
