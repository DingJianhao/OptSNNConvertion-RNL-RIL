import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
import torch.optim as optim
import models
import argparse
import json
import modules
from utils import *
import numpy as np
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith('__') and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--pretrain', default='train_vgg16_cifar10_demo', help='Model dir.')

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Choice of optimizer.')
parser.add_argument('--epoch', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01, help='Global learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout applied to the model.')

parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7', 'cuda:8', 'cuda:9', 'cuda:10']
                    , help='Device.')

parser.add_argument('--lam', type=float, default=0.1, help='multiplied with omega')
parser.add_argument('--acc_tolerance', type=float, default=0.03, help='accuracy tolerance')
parser.add_argument('--sharescale', default=False, action='store_true', help='Disable scale share.')
parser.add_argument('--suffix', type=str, default='', help='model save name suffix')

parser.add_argument('--init', type=float, default=4, help='init scale.')

args = parser.parse_args()
args.args = os.path.join(args.pretrain,'config.json')
with open(args.args, 'r') as fp:
    d = json.load(fp)

args.dataset = d['dataset']
args.no_data_aug = d['no_data_aug']
args.model = d['model']
args.log_dir = d['log_dir']
log_dir = args.log_dir
args.load_name = os.path.join(args.log_dir, d['save_name'] + args.suffix + '.pth')
args.save_name = d['save_name']

with open(os.path.join(log_dir, 'config2.json'), 'w') as fw:
    json.dump(vars(args), fw)

args.num_classes = num_classes[args.dataset]
args.device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
assert len(args.gammas) == len(args.schedule)
print(args.log_dir)
print_args(args)

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
with open(os.path.join(log_dir, 'config.json'), 'w') as fw:
    json.dump(vars(args), fw)
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                                                 batch_size=args.batch_size,
                                                 workers=0,
                                                 dataset=args.dataset,
                                                 data_target_dir=datapath[args.dataset]
                                                 )
best_acc = 0.0
best_avg_k = 1e5
start_epoch = 0
sum_k = 0.0
cnt_k = 0.0
train_batch_cnt = 0
test_batch_cnt = 0
model = models.__dict__[args.model](num_classes=args.num_classes, dropout=args.dropout)

model = modules.replace_maxpool2d_by_avgpool2d(model)
model = modules.replace_relu_by_spikingnorm(model, True)

if args.load_name and os.path.isfile(args.load_name):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.load_name)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, val=1)
            nn.init.zeros_(m.bias)
model.to(args.device)
args.device = torch.device(args.device)
if args.device.type == 'cuda':
    print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(args.device.index)}")


if args.sharescale:
    first_scale = None
    sharescale = Parameter(torch.Tensor([args.init])) # for vgg16: 2.5                 for resnet ??5
    for m in model.modules():
        if isinstance(m, modules.SpikingNorm) and first_scale is None:
            first_scale = m.scale
        elif isinstance(m, modules.SpikingNorm) and first_scale is not None:
            setattr(m, 'scale', first_scale)
        if isinstance(m, modules.SpikingNorm):
            setattr(m, 'scale', sharescale)
else:
    for m in model.modules():
        if isinstance(m, modules.SpikingNorm):
            m.scale.data *= args.init / m.scale.data

ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()


def divide_trainable_modules(model):
    global ann_train_module, snn_train_module
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = divide_trainable_modules(module)
        if module.__class__.__name__ != "Sequential":
            if module.__class__.__name__ == "SpikingNorm":
                snn_train_module.append(module)
            else:
                ann_train_module.append(module)
    return model


divide_trainable_modules(model)


def new_loss_function(ann_out, snn_out, k, func='cos'):
    if func == 'mse':
        f = nn.MSELoss()
        diff_loss = f(ann_out, snn_out)
    elif func == 'cos':
        f = nn.CosineSimilarity(dim=1, eps=1e-6)
        diff_loss = 1.0 - torch.mean(f(ann_out, snn_out))
    else:
        assert False
    loss = diff_loss + args.lam * k
    return loss, diff_loss


loss_function1 = nn.CrossEntropyLoss()
loss_function2 = new_loss_function

if args.optimizer == 'sgd':
    optimizer2 = optim.SGD(snn_train_module.parameters(),
                           momentum=args.momentum,
                           lr=args.lr,
                           weight_decay=args.decay)
elif args.optimizer == 'adam':
    optimizer2 = optim.Adam(snn_train_module.parameters(),
                           lr=args.lr,
                           weight_decay=args.decay)

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir + '/' + 'runs_' + current_time)


def adjust_learning_rate(optimizer, epoch):
    global args
    lr = args.lr
    for (gamma, step) in zip(args.gammas, args.schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


sum_k = 0
cnt_k = 0
last_k = 0
test_batch_cnt = 0
train_batch_cnt = 0


def layerwise_k(a, max=1.0):
    return torch.sum(a / max) / (torch.pow(torch.norm(a / max, 2), 2) + 1e-5)


def hook(module, input, output):
    global sum_k, cnt_k
    sum_k += layerwise_k(output)
    cnt_k += 1
    return


def snn_train(epoch, args):
    global sum_k, cnt_k, train_batch_cnt, last_k
    net = model.to(args.device)

    print('\nEpoch: %d Fast Train' % epoch)
    net.train()
    snn_fast_loss = 0
    snn_dist_loss = 0
    snn_correct = 0
    total = 0

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(hook))

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        sum_k = 0
        cnt_k = 0
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        ann_outputs = net(inputs)
        ann_loss = loss_function1(ann_outputs, targets)

        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('encounter ann_loss', ann_loss)
            return False

        predict_outputs = ann_outputs.detach()
        _, ann_predicted = predict_outputs.max(1)

        snn_outputs = net(inputs)
        last_k = layerwise_k(F.relu(snn_outputs), torch.max(snn_outputs))
        fast_loss, dist_loss = loss_function2(predict_outputs, snn_outputs, (sum_k + last_k) / (cnt_k + 1))
        snn_dist_loss += dist_loss.item()
        snn_fast_loss += fast_loss.item()
        optimizer2.zero_grad()
        fast_loss.backward()
        optimizer2.step()

        _, snn_predicted = snn_outputs.max(1)
        tot = targets.size(0)
        total += tot
        sc = snn_predicted.eq(targets).sum().item()
        snn_correct += sc

        writer.add_scalar('Train/Acc', sc / tot, train_batch_cnt)
        writer.add_scalar('Train/DistLoss', dist_loss, train_batch_cnt)
        writer.add_scalar('Train/AvgK', (sum_k / cnt_k).item(), train_batch_cnt)
        writer.add_scalar('Train/LastK', last_k, train_batch_cnt)
        train_batch_cnt += 1
    print('Fast Train Epoch %d Loss:%.3f Acc:%.3f' % (epoch,
                                                      snn_dist_loss,
                                                      snn_correct / total))

    writer.add_scalar('Train/EpochAcc', snn_correct / total, epoch)
    for handle in handles:
        handle.remove()
    return True


def get_acc():
    global model
    net = model
    net.to(args.device)

    net.eval()
    correct = 0
    total = 0
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            m.snn = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    snn_acc = correct / total
    return snn_acc


def val(epoch, args):
    global sum_k, cnt_k, test_batch_cnt, best_acc, last_k, best_avg_k
    net = model.to(args.device)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            ann_outputs = net(inputs)
            ann_loss = loss_function1(ann_outputs, targets)

            if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
                print('encounter ann_loss', ann_loss)
                return False

            predict_outputs = ann_outputs.detach()
            ann_test_loss += (ann_loss.item())
            _, ann_predicted = predict_outputs.max(1)

            tot = targets.size(0)
            total += tot
            ac = ann_predicted.eq(targets).sum().item()
            ann_correct += ac

            last_k = layerwise_k(F.relu(ann_outputs), torch.max(ann_outputs))
            writer.add_scalar('Test/Acc', ac / tot, test_batch_cnt)
            writer.add_scalar('Test/Loss', ann_test_loss, test_batch_cnt)
            writer.add_scalar('Test/AvgK', (sum_k / cnt_k).item(), test_batch_cnt)
            writer.add_scalar('Test/LastK', last_k, test_batch_cnt)
            test_batch_cnt += 1
        print('Test Epoch %d Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (epoch,
                                                                         ann_test_loss,
                                                                         ann_correct / total,
                                                                         sum_k / cnt_k, last_k))
    writer.add_scalar('Test/EpochAcc', ann_correct / total, epoch)

    # Save checkpoint.
    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    acc = 100. * ann_correct / total
    if acc > (best_acc - args.acc_tolerance)*100. and best_avg_k > avg_k:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'avg_k': avg_k
        }
        if not os.path.isdir(args.log_dir):
            os.mkdir(args.log_dir)
        torch.save(state, args.log_dir + '/%s_[%.3f_%.3f_%.3f].pth' % (args.save_name,
                                                                       args.lam,acc,
                                                                       ((sum_k + last_k) / (cnt_k + 1)).item() ))
        best_avg_k = avg_k

    if (epoch + 1) % 10 == 0:
        print('Schedule Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.log_dir + '/%s_ft_scheduled.pth' % (args.save_name))
    for handle in handles:
        handle.remove()


# print(model)
best_acc = get_acc()
print('pretrain: ',best_acc)
# exit(-1)
for epoch in range(start_epoch, start_epoch + args.epoch):
    adjust_learning_rate(optimizer2, epoch)
    ret = snn_train(epoch, args)
    if ret == False:
        exit(-1)
    val(epoch, args)
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data, 'scale_t',m.scale.data)
