import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import models
import json
import modules
from utils import *
import numpy as np
import torch.nn.functional as F
import spikingjelly.clock_driven.functional as functional
import matplotlib.pyplot as plt
import spikingjelly.clock_driven.neuron as neuron

####################################################
#
# Model init
#
####################################################
model_name = 'vgg16'
dataset = 'cifar10'
device = 'cuda'
optimizer = 'sgd'

momentum = 0.9
lr = 0.1
schedule = [100, 150]
gammas = [0.1, 0.1]
decay = 1e-4
batch_size = 50
epoch = 200
acc_tolerance = 0.1
lam = 0.1
sharescale = True
scale_init = 2.5
conf = [model_name,dataset]
save_name = '_'.join(conf)
log_dir = 'train_' + save_name

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=batch_size,
                 workers=0,
                 dataset=dataset,
                 data_target_dir=datapath[dataset]
                 )

best_acc = 0.0
start_epoch = 0
sum_k = 0.0
cnt_k = 0.0
train_batch_cnt = 0
test_batch_cnt = 0
model = models.__dict__[model_name](num_classes=10, dropout=0)

model = modules.replace_maxpool2d_by_avgpool2d(model)
model = modules.replace_relu_by_spikingnorm(model,True)

for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m,'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, val=1)
        nn.init.zeros_(m.bias)
model.to(device)
device = torch.device(device)
if device.type == 'cuda':
    print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(device.index)}")


ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()

def divide_trainable_modules(model):
    global ann_train_module,snn_train_module
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
    loss = diff_loss + lam * k
    return loss, diff_loss

loss_function1 = nn.CrossEntropyLoss()
loss_function2 = new_loss_function

# define opt1
if optimizer == 'sgd':
    optimizer1 = optim.SGD(ann_train_module.parameters(),
                               momentum=momentum,
                               lr=lr,
                               weight_decay=decay)
elif optimizer == 'adam':
    optimizer1 = optim.Adam(ann_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
elif optimizer == 'adamw':
    optimizer1 = optim.AdamW(ann_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)

writer = SummaryWriter(log_dir)

###################################################
#
# some function
#
###################################################

def adjust_learning_rate(optimizer, epoch):
    global lr
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

sum_k = 0
cnt_k = 0
last_k = 0
best_avg_k = 1e5
test_batch_cnt = 0
train_batch_cnt = 0

def layerwise_k(a, max=1.0):
    return torch.sum(a / max) / (torch.pow(torch.norm(a / max, 2), 2) + 1e-5)

def hook(module, input, output):
    global sum_k,cnt_k
    sum_k += layerwise_k(output)
    cnt_k += 1
    return

def ann_train(epoch):
    global sum_k,cnt_k,train_batch_cnt
    net = model.to(device)

    print('\nEpoch: %d Para Train' % epoch)
    net.train()
    ann_train_loss = 0
    ann_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        ann_outputs = net(inputs)
        ann_loss = loss_function1(ann_outputs, targets)

        ann_train_loss += (ann_loss.item())
        _, ann_predicted = ann_outputs.max(1)

        tot = targets.size(0)
        total += tot
        ac = ann_predicted.eq(targets).sum().item()
        ann_correct += ac

        optimizer1.zero_grad()
        ann_loss.backward()
        # torch.nn.utils.clip_grad_norm_(ann_train_module.parameters(), 50)
        optimizer1.step()
        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('encounter ann_loss', ann_loss)
            return False

        writer.add_scalar('Train/Acc', ac / tot, train_batch_cnt)
        writer.add_scalar('Train/Loss', ann_loss.item(), train_batch_cnt)
        train_batch_cnt += 1
    print('Para Train Epoch %d Loss:%.3f Acc:%.3f' % (epoch,
                                                      ann_train_loss,
                                                      ann_correct / total))
    writer.add_scalar('Train/EpochAcc', ann_correct / total, epoch)
    return

def para_train_val(epoch):
    global sum_k,cnt_k,test_batch_cnt,best_acc
    net = model.to(device)

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
            inputs, targets = inputs.to(device), targets.to(device)
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
    acc = 100.*ann_correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        torch.save(state, log_dir + '/%s.pth'%(save_name))
        best_acc = acc

    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    if (epoch + 1) % 10 == 0:
        print('Schedule Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'avg_k': avg_k
        }
        torch.save(state, log_dir + '/%s_pt_scheduled.pth' % (save_name))
    for handle in handles:
        handle.remove()

def snn_train(epoch):
    global sum_k, cnt_k, train_batch_cnt, last_k
    net = model.to(device)

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
        inputs, targets = inputs.to(device), targets.to(device)
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
        if train_batch_cnt % inspect_interval == 0:
            if not snn_val(train_batch_cnt):
                return False
            net.train()
    print('Fast Train Epoch %d Loss:%.3f Acc:%.3f' % (epoch,
                                                      snn_dist_loss,
                                                      snn_correct / total))

    writer.add_scalar('Train/EpochAcc', snn_correct / total, epoch)
    for handle in handles:
        handle.remove()
    return True

def get_acc(val_dataloader):
    global model
    net = model
    net.to(device)

    net.eval()
    correct = 0
    total = 0
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            m.lock_max = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    snn_acc = correct / total
    return snn_acc

def snn_val(iter):
    global sum_k, cnt_k, test_batch_cnt, best_acc, last_k, best_avg_k
    net = model.to(device)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(device), targets.to(device)
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
        print('Test Iter %d Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (iter,
                                                                         ann_test_loss,
                                                                         ann_correct / total,
                                                                         sum_k / cnt_k, last_k))
    writer.add_scalar('Test/IterAcc', ann_correct / total, iter)

    # Save checkpoint.
    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    acc = 100. * ann_correct / total
    if acc < (best_acc - acc_tolerance)*100.:
        return False
    if acc > (best_acc - acc_tolerance)*100. and best_avg_k > avg_k:
        test_acc = get_acc(test_dataloader)
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc * 100,
            'epoch': epoch,
            'avg_k': avg_k
        }
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        torch.save(state, log_dir + '/%s_[%.3f_%.3f_%.3f].pth' % (save_name,
                                                                       lam,test_acc * 100,
                                                                       ((sum_k + last_k) / (cnt_k + 1)).item() ))
        best_avg_k = avg_k

    if (epoch + 1) % 10 == 0:
        print('Schedule Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, log_dir + '/%s_ft_scheduled.pth' % (save_name))
    for handle in handles:
        handle.remove()
    return True

def simulate(net, T, save_name, log_dir, ann_baseline=0.0):
    net.to(device)
    functional.reset_net(net)
    correct_t = {}

    with torch.no_grad():
        net.eval()
        correct = 0.0
        total = 0.0
        for batch, (img, label) in enumerate(test_dataloader):
            for t in range(T):
                out = net(img.to(device))
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                if t == 0:
                    out_spikes_counter = out
                else:
                    out_spikes_counter += out
                if t not in correct_t.keys():
                    correct_t[t] = (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                else:
                    correct_t[t] += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            correct += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            total += label.numel()
            functional.reset_net(net)

            fig = plt.figure()
            x = np.array(list(correct_t.keys())).astype(np.float32) + 1
            y = np.array(list(correct_t.values())).astype(np.float32) / total * 100
            plt.plot(x, y, label='SNN', c='b')
            if ann_baseline != 0:
                plt.plot(x, np.ones_like(x) * ann_baseline, label='ANN', c='g', linestyle=':')
                plt.text(0, ann_baseline + 1, "%.3f%%" % (ann_baseline), fontdict={'size': '8', 'color': 'g'})
            plt.title("%s Simulation \n[test samples:%.1f%%]" % (
                save_name, 100 * total / len(test_dataloader.dataset)
            ))
            plt.xlabel("T")
            plt.ylabel("Accuracy(%)")
            plt.legend()
            argmax = np.argmax(y)
            disp_bias = 0.3 * float(T) if x[argmax] / T > 0.7 else 0
            plt.text(x[argmax] - 0.8 - disp_bias, y[argmax] + 0.8, "MAX:%.3f%% T=%d" % (y[argmax], x[argmax]),
                     fontdict={'size': '12', 'color': 'r'})

            plt.scatter([x[argmax]], [y[argmax]], c='r')
            print('[SNN Simulating... %.2f%%] Acc:%.3f' % (100 * total / len(test_dataloader.dataset),
                                                                     correct / total))
            acc_list = np.array(list(correct_t.values())).astype(np.float32) / total * 100
            np.save(log_dir + '/snn_acc-list' + ('-constant'), acc_list)
            plt.savefig(log_dir + '/sim_' + save_name + ".jpg", dpi=1080)

            from PIL import Image
            im = Image.open(log_dir + '/sim_' + save_name + ".jpg")
            totensor = transforms.ToTensor()
            plt.close()
        acc = correct / total
        print('SNN Simulating Accuracy:%.3f' % (acc ))

def replace_spikingnorm_by_ifnode(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_spikingnorm_by_ifnode(module)
        if module.__class__.__name__ == "SpikingNorm":
            model._modules[name] = neuron.IFNode(v_threshold=module.calc_v_th().data.item(),v_reset=None)
    return model

def simulate_by_filename(save_name):
    model = models.__dict__[model_name](num_classes=10, dropout=0)
    model = modules.replace_maxpool2d_by_avgpool2d(model)
    model = modules.replace_relu_by_spikingnorm(model,True)
    state_dict = torch.load('train_vgg16_cifar10/%s.pth' % save_name)
    ann_acc = state_dict['acc']
    model.load_state_dict(state_dict['net'])
    model = replace_spikingnorm_by_ifnode(model)
    simulate(model.to(device), T=100, save_name='%s' % save_name, log_dir=log_dir, ann_baseline=ann_acc)

####################################################
#
# Phase 1 training: training for weight parameter
#
####################################################

for epoch in range(start_epoch, start_epoch + epoch):
    adjust_learning_rate(optimizer1, epoch)
    if epoch==start_epoch:
        para_train_val(epoch)
    ret = ann_train(epoch)
    if ret == False:
        break
    para_train_val(epoch)
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data)

####################################################
#
# Phase 2 training: training for fast inference
#
####################################################

dataset = train_dataloader.dataset
train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model.load_state_dict(torch.load('train_vgg16_cifar10/vgg16_cifar10.pth')['net'])

if sharescale:
    first_scale = None
    sharescale = nn.Parameter(torch.Tensor([scale_init]))
    for m in model.modules():
        if isinstance(m, modules.SpikingNorm):
            setattr(m, 'scale', sharescale)
            m.lock_max = True

divide_trainable_modules(model)

# define opt2
lr = 0.001
inspect_interval = 100
if optimizer == 'sgd':
    optimizer2 = optim.SGD(snn_train_module.parameters(),
                           momentum=momentum,
                           lr=lr,
                           weight_decay=decay)
elif optimizer == 'adam':
    optimizer2 = optim.Adam(snn_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)

best_acc = get_acc(val_dataloader)
for e in range(0, epoch):
    adjust_learning_rate(optimizer2, e)
    ret = snn_train(e)
    if ret == False:
        break
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data, 'scale_t',m.scale.data)

####################################################
#
# Simulate model
#
####################################################

simulate_by_filename('vgg16_cifar10_[0.100_87.880_7.643]')
simulate_by_filename('vgg16_cifar10_[0.100_86.840_6.528]')
simulate_by_filename('vgg16_cifar10_[0.100_84.440_5.808]')

