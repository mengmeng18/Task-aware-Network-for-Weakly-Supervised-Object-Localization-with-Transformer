import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    w_list = []
    b_list = []
    last_weight_list = []
    last_bias_list =[]
    for name, value in model.named_parameters():
        if 'part' in name or 'mask' in name or 'head' in name:   # or 'memory' in name
            print(name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    weight_decay = 0.0001

    opt = SGD_GCC([{'params': weight_list, 'lr': lr},
                     {'params': bias_list, 'lr': lr*2},
                     {'params': last_weight_list, 'lr': lr*10},
                     {'params': last_bias_list, 'lr':  lr*20}], momentum=0.9, weight_decay=0.0005, nesterov=True)
    return opt

def optimizer_G(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    for name, value in model.named_parameters():
        if 'memory' in name:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    weight_decay = 0.0001
    opt = SGD_GCC([{'params': weight_list, 'lr': lr},
                   {'params': bias_list, 'lr': lr * 2}
                   ],
                  momentum=0.9, weight_decay=0.0005, nesterov=True)
    return opt

def optimizer_D(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if 'part' in name and 'fc' not in name:
            # print(name)
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)
        if 'binary_cls' in name:
            print(name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)

    weight_decay = 0.0001
    opt = SGD_GCC([{'params': weight_list, 'lr': lr},
                    {'params': bias_list, 'lr': lr * 2},
                   {'params': last_weight_list, 'lr': lr * 10},
                   {'params': last_bias_list, 'lr': lr * 20}
                   ],
                  momentum=0.9, weight_decay=0.0005, nesterov=True)
    return opt

def lr_poly(base_lr, iter,max_iter,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)

def get_optimizer(args, model):
    lr = args.lr
    # opt = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    opt = optim.SGD(params=[para for name, para in model.named_parameters() if 'features' not in name], lr=lr, momentum=0.9, weight_decay=0.0001)
    # lambda1 = lambda epoch: 0.1 if epoch in [85, 125, 165] else 1.0
    # scheduler = LambdaLR(opt, lr_lambda=lambda1)

    return opt

def get_adam(args, model):
    lr = args.lr
    opt = optim.Adam(params=model.parameters(), lr =lr, weight_decay=0.0005)
    # opt = optim.Adam(params=model.parameters(), lr =lr)

    return opt

def reduce_lr(args, optimizer, epoch, factor=0.1):
    if 'coco' in args.dataset:
        change_points = [1,2,3,4,5]
    elif 'imagenet' in args.dataset:
        change_points = [1,2,3,4,5,6,7,8,9,10,11,12]
    else:
        change_points = None

    # values = args.decay_points.strip().split(',')
    # try:
    #     change_points = map(lambda x: int(x.strip()), values)
    # except ValueError:
    #     change_points = None

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            print(epoch, g['lr'])
        return True

def adjust_lr(args, optimizer, epoch):
    if 'cifar' in args.dataset:
        change_points = [80, 120, 160]
    elif 'indoor' in args.dataset:
        change_points = [60, 80, 100]
    elif 'dog' in args.dataset:
        change_points = [60, 80, 100]
    elif 'voc' in args.dataset:
        change_points = [30, 40]
    else:
        change_points = None
    # else:

    # if epoch in change_points:
    #     lr = args.lr * 0.1**(change_points.index(epoch)+1)
    # else:
    #     lr = args.lr

    if change_points is not None:
        change_points = np.array(change_points)
        pos = np.sum(epoch > change_points)
        lr = args.lr * (0.1**pos)
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class SGD_GCC(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_GCC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_GCC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # GC operation for Conv layers
                if len(list(d_p.size())) > 3:
                    d_p.add_(-d_p.mean(dim=tuple(range(1, len(list(d_p.size())))), keepdim=True))

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss