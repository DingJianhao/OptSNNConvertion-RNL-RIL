import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import modules as modules
import models
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
import spikingjelly.clock_driven.functional as functional
import spikingjelly.clock_driven.encoding as encoding
import matplotlib.pyplot as plt
import spikingjelly.clock_driven.neuron as neuron
import copy
from collections import OrderedDict,defaultdict
import tqdm
import json
import torch.nn.functional as F
from utils import *

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--pretrain', default='train_vgg16_cifar10_demo2', help='Model dir.')
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3',
                             'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7', 'cuda:8', 'cuda:9', 'cuda:10']
                    , help='Device.')
parser.add_argument('--suffix', type=str, default='', help='model save name suffix')

parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
parser.add_argument('--simulation', '-sim', type=str, default='acc', choices=['acc', 'curve', 'power'],help='simulate')
parser.add_argument('--poisson', default=False, help='if poisson.')
parser.add_argument('--T', type=int, default=500, help='time step.')
parser.add_argument('--prefix', type=str, default='', help='model dir prefix')
parser.add_argument('--loadmodel', default=False, action='store_true', help='if loadmodel.')


args = parser.parse_args()
args.args = os.path.join(args.pretrain,'config.json')
with open(args.args, 'r') as fp:
    d = json.load(fp)

args.dataset = d['dataset']
args.no_data_aug = d['no_data_aug']
args.model = d['model']
args.load_name = os.path.join(d['log_dir'], d['save_name'] + args.suffix + '.pth')
args.save_name = d['save_name']

if len(args.prefix) == 0:
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = 'evalu_' + args.save_name + '_' + current_time
else:
    log_dir = 'evalu_' + args.save_name + '_' + args.prefix
args.log_dir = log_dir

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
with open(os.path.join(log_dir, 'config.json'), 'w') as fw:
    json.dump(vars(args), fw)

args.num_classes = num_classes[args.dataset]
args.device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
print(args.log_dir)
print_args(args)

train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=args.batch_size,
                 workers=0,
                 dataset=args.dataset,
                 data_target_dir=datapath[args.dataset]
                 )

model = models.__dict__[args.model](num_classes=args.num_classes, dropout=d['dropout'])

model = modules.replace_maxpool2d_by_avgpool2d(model)
model = modules.replace_relu_by_spikingnorm(model,True)

if args.load_name and os.path.isfile(args.load_name):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.load_name)
    if not args.loadmodel:
        model.load_state_dict(checkpoint['net'])
    else:
        model = checkpoint['model']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    assert False,"no checkpoint found! %s" % args.load_name
model.to(args.device)
model.eval()
args.device = torch.device(args.device)
if args.device.type == 'cuda':
    print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(args.device.index)}")


def get_acc():
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

def simulate(ann_baseline=0.0):
    net = model
    poisson = args.poisson
    save_name = args.save_name
    T = args.T
    net.to(args.device)
    log_dir = args.log_dir
    writer = SummaryWriter(args.log_dir)

    functional.reset_net(net)
    if poisson:
        encoder = encoding.PoissonEncoder()
    correct_t = {}

    with torch.no_grad():
        net.eval()
        correct = 0.0
        total = 0.0
        for batch, (img, label) in enumerate(test_dataloader):
            img = img.to(args.device)
            for t in tqdm.tqdm(range(T)):
                encoded = encoder(img).float() if poisson else img
                out = net(encoded)
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                if t == 0:
                    out_spikes_counter = out
                else:
                    out_spikes_counter += out
                if t not in correct_t.keys():
                    correct_t[t] = (out_spikes_counter.max(1)[1] == label.to(args.device)).float().sum().item()
                else:
                    correct_t[t] += (out_spikes_counter.max(1)[1] == label.to(args.device)).float().sum().item()
            correct += (out_spikes_counter.max(1)[1] == label.to(args.device)).float().sum().item()
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
            np.save(log_dir + '/snn_acc-list' + ('-poisson' if poisson else '-constant'), acc_list)
            plt.savefig(log_dir + '/sim_' + save_name + ".jpg", dpi=1080)

            from PIL import Image
            im = Image.open(log_dir + '/sim_' + save_name + ".jpg")
            totensor = transforms.ToTensor()
            writer.add_image('simulation', totensor(im), 0)
            plt.close()
        acc = correct / total
        print('SNN Simulating Accuracy:%.3f' % (acc ))

    writer.close()

def layerwise_k(a, max=1.0):
    return torch.sum(a / max) / (torch.pow(torch.norm(a / max, 2), 2) + 1e-5)

global_idx = 0
omega = []
ann_output = OrderedDict()
sum_spk_trains = defaultdict(float)
dist_to_ann = defaultdict(list)
writer = SummaryWriter(log_dir)

def converge(a, b):
    m1 = np.linalg.norm(a.reshape(-1) - b.reshape(-1), 2) ** 2
    m2 = np.linalg.norm(b.reshape(-1), 2) ** 2
    return m1 / m2

def spknorm_forward_hook(module, input, output):
    global global_idx, ann_output
    ann_output[global_idx] = output.detach().cpu().numpy()
    norm = layerwise_k(output)
    print('layer', global_idx, 'sum(a)/[norm(a,2)^2]', norm.item())
    omega.append(norm.item())
    global_idx += 1

def ifnode_forward_hook(module, input, output):
    global global_idx, ann_output, sum_spk_trains, writer
    t = len(module.monitor['s'])
    sum_spk_trains[global_idx] += module.monitor['s'][-1]
    spk_rate = sum_spk_trains[global_idx] / t
    v = converge(spk_rate, ann_output[global_idx])
    dist_to_ann[global_idx].append(v)
    writer.add_scalar('Curve/l%d' % global_idx, v, t-1)
    global_idx += 1


def simulate_curve(ann,data,data_label,T=1000,poisson=False):
    global writer, sum_spk_trains, ann_output, dist_to_ann, omega, global_idx
    total = data_label.size(0)

    ann.eval()
    plt.ion()
    print(data.size())
    snn = copy.deepcopy(ann)
    snn.eval()

    snn = modules.replace_spikingnorm_by_ifnode(snn)
    snn = snn.to(args.device)
    encoder = encoding.PoissonEncoder()

    ann_output['input'] = data.detach().cpu().numpy()

    for m in ann.modules():
        if isinstance(m, modules.SpikingNorm):
            m.lock_max = True

    global_idx = 0
    x = data

    handles = []
    for m in ann.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(spknorm_forward_hook))
    x = ann(x)
    for handle in handles:
        handle.remove()

    print('layer', global_idx, 'sum(a)/[norm(a,2)^2]', layerwise_k(F.relu(x),torch.max(x).item()).item())
    omega.append(layerwise_k(F.relu(x),torch.max(x).item()).item())
    ann_output[global_idx] = x.detach().cpu().numpy()

    np.savetxt(os.path.join(log_dir,"omegas.csv"), omega, delimiter=",")

    for n, m in snn.named_modules():
        if "IFNode" in m.__class__.__name__:
            m.monitor = {'v': [], 's': []}
    functional.reset_net(snn)
    output_acc = []
    sum_input_spks = 0
    output_sum = 0

    handles = []
    for m in snn.modules():
        if isinstance(m, neuron.IFNode):
            handles.append(m.register_forward_hook(ifnode_forward_hook))

    for t in tqdm.tqdm(range(T)):
        plt.cla()
        x = encoder(data).float() if poisson else data
        sum_input_spks += x.detach().cpu().numpy()

        global_idx = 0
        output = snn(x)
        output_sum += output

        v = torch.argmax(output_sum, dim=1).eq(data_label).sum().item() / total
        output_acc.append(v)
        writer.add_scalar('Curve/acc', v, t)
        v = converge(sum_input_spks / (t + 1),ann_output['input'])
        dist_to_ann['input'].append(v)
        writer.add_scalar('Curve/i', v ,t)

        sum_output_spks = output_sum.detach().cpu().numpy()
        v = converge(sum_output_spks / (t + 1), ann_output[global_idx])
        dist_to_ann[global_idx].append(v)
        writer.add_scalar('Curve/o', v, t)

        x_lst = np.arange(1, t + 2)
        plt.scatter(x_lst, output_acc, c='r', marker='x', label='Acc')
        for i in dist_to_ann.keys():
            if not poisson and i == 'input':
                continue
            lst = np.array(dist_to_ann[i])
            plt.plot(x_lst, lst, label=i)

        plt.ylim([0,1.0])
        plt.legend(prop={'size': 6})
        plt.pause(0.01)
        plt.savefig(log_dir + "\single_pic_infer%s.pdf" % ('-poisson' if poisson else '-constant'))
    plt.savefig(log_dir + "\single_pic_infer%s.pdf" % ('-poisson' if poisson else '-constant'))
    plt.ioff()
    plt.close()

    for handle in handles:
        handle.remove()

    for n, m in snn.named_modules():
        if "IFNode" in m.__class__.__name__:
            m.monitor = False
    pass

def simulate_power(ann,data,T=1000):
    poisson = False
    '''
    total_spikes.csv
    (T) layer1 layer2 layer3 layer4 ...
     0    100    200    122    324
     1    100    200    122    324
    ...

    total_neurons.csv
    layer1  layer2 ...
      100     200
    
    data_num.csv
    50
    '''
    total = data.size(0)
    ann = ann.to(args.device)
    ann.eval()
    snn = copy.deepcopy(ann)
    snn.eval()
    snn = modules.replace_spikingnorm_by_ifnode(snn)

    data = data.to(args.device)
    x = ann(data)
    outnode = neuron.IFNode(v_threshold=torch.max(x).item(),v_reset=None)
    snn = nn.Sequential(snn, outnode)
    snn.to(args.device)
    functional.reset_net(snn)

    encoder = encoding.PoissonEncoder()
    sum_outputs = 0.0

    if_list = []
    for n, m in snn.named_modules():
        if "IFNode" in m.__class__.__name__:
            m.monitor = {'v': [], 's': []}
            if_list.append(m)

    for t in tqdm.tqdm(range(T)):
        x = encoder(data).float() if poisson else data
        output = snn(x)

    layer_list = []
    layer_numel_list = []
    for m in tqdm.tqdm(if_list):
        layer_numel_list.append(m.monitor['s'][0].reshape(-1).shape[0])
        spks_list = []
        for spks in m.monitor['s']:
            spks_list.append(np.sum(spks))
        layer_list.append(spks_list)

    np.savetxt(os.path.join(log_dir, "data_num.csv"), np.array([total]), delimiter=",")
    np.savetxt(os.path.join(log_dir, "total_spikes.csv"), np.array(layer_list).transpose(), delimiter=",")
    np.savetxt(os.path.join(log_dir, "total_neurons.csv"), np.array(layer_numel_list), delimiter=",")


acc = get_acc()
print('best acc:',acc)
if args.simulation == 'acc':
    model = modules.replace_spikingnorm_by_ifnode(model)
    print(model)
    simulate(acc*100)
elif args.simulation == 'curve':
    test_dataloader = iter(test_dataloader)
    data,target = next(test_dataloader)
    data = data.to(args.device)
    target = target.to(args.device)
    # data = test_dataloader.dataset.data[10, :, :].astype(np.float32) / 255
    # data_label = test_dataloader.dataset.targets[10]
    # data = torch.from_numpy(data.reshape(1, 32, 32, 3).transpose(0, 3, 1, 2).astype(np.float32))
    # data = data.to(args.device)
    # label = torch.tensor(data_label)
    with torch.no_grad():
        simulate_curve(model, data, target, T=args.T, poisson=False)
    # else:
    #     assert False, "unable to sim curve!"

elif args.simulation == 'power':
    test_dataloader = iter(test_dataloader)
    data, target = next(test_dataloader)
    data = data.to(args.device)
    target = target.to(args.device)
    with torch.no_grad():
        simulate_power(model, data, T=args.T)
