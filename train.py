from __future__ import print_function

import argparse
import logging

import git
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data as torch_data

from data.dataset import Dataset
from models import resnet, metrics, fmobilefacenet
from models.focal_loss import FocalLoss
from test import *
from utils.visualizer import Visualizer


def save_model(model, metric_fc, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_base_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    save_name = os.path.join(save_path, name + '_weight_' + str(iter_cnt) + '.pth')
    torch.save(metric_fc.state_dict(), save_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')

    leveldb_path = os.path.expanduser("/opt/cacher/faces_webface_112x112")
    parser.add_argument('--leveldb_path', default=leveldb_path, help='training set directory')
    label_path = os.path.expanduser("/opt/cacher/faces_webface_112x112.labels")
    parser.add_argument('--label_path', default=label_path, help='training set directory')

    target = os.path.expanduser("~/datasets/maysa/lfw.bin")
    parser.add_argument('--target', type=str, default=target, help='verification targets')

    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in each context')
    parser.add_argument('--num_workers', type=int, default=2, help='batch size in each context')
    parser.add_argument('--loss', type=str, default="focal_loss", help='batch size in each context')
    parser.add_argument('--metric', type=str, default="arc_margin", help='batch size in each context')

    parser.add_argument('--pretrained', default='', help='pretrained model to load')
    parser.add_argument('--network', default='resnet18', help='specify network')
    parser.add_argument('--optimizer', default='sgd', help='specify network')
    parser.add_argument('--margin_s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin_m', type=float, default=0.5, help='margin for loss,')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')

    parser.add_argument('--lr_steps', type=str, default='8, 12, 16', help='steps of lr changing')
    parser.add_argument('--use_se', default=False, action='store_true', help='if output ce loss')
    parser.add_argument('--easy_margin', default=False, action='store_true', help='')
    parser.add_argument('--display', default=False, action='store_true', help='if output ce loss')
    parser.add_argument('--print_freq', type=int, default=100, help='重新加载feature')

    parser.add_argument('--emb_size', type=int, default=512, help='embedding length')
    args = parser.parse_args()
    return args


def train_net(args):
    branch_name = git.Repo(".").active_branch.name
    prefix = time.strftime("%Y-%m-%d-%H:%M:%S")
    file_path = "train/{}_{}".format(branch_name, prefix)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler("{}/train.log".format(file_path))
    # create formatter#
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # add formatter to ch
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    args.pretrained = os.path.expanduser(args.pretrained)

    if args.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(args.leveldb_path, args.label_path)
    trainloader = torch_data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)

    logging.info('{} train iters per epoch'.format(len(trainloader)))

    if args.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.network == 'resnet18':
        model = fmobilefacenet.resnet_face18(args=args)
    elif args.network == 'resnet34':
        model = resnet.resnet34()
    elif args.network == 'resnet50':
        model = resnet.resnet50()

    num_classes = train_dataset.label_len
    if args.metric == 'add_margin':
        metric_fc = metrics.AddMarginProduct(512, num_classes, s=args.margin_s, m=0.35)
    elif args.metric == 'arc_margin':
        metric_fc = metrics.ArcMarginProduct(512, num_classes, s=args.margin_s, m=args.margin_m, easy_margin=args.easy_margin)
    elif args.metric == 'sphere':
        metric_fc = metrics.SphereProduct(512, num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, args.num_classes)

    # view_model(model, opt.input_shape)
    if args.pretrained:
        model.load_state_dict()
        metric_fc.load_state_dict()
    # logging.info(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr, weight_decay=args.weight_decay)

    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    max_epoch = 2 * lr_steps[-1] - lr_steps[-2]
    start = time.time()
    for i in range(max_epoch):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            cosine, output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % args.print_freq == 0:
                label = label.data.cpu().numpy()

                cosine = cosine.data.cpu().numpy()
                cosine = np.argmax(cosine, axis=1)
                real_acc = np.mean((cosine == label).astype(int))


                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                acc = np.mean((output == label).astype(int))

                speed = args.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                logging.info('time %s train lr %s epoch %s iter/size %s/%s iters %s %s iters/s loss %s acc %s real_acc %s',
                             time_str, scheduler.get_lr(), i, ii, len(trainloader), iters, speed, loss.item(), acc, real_acc)

                if args.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        save_model(model, metric_fc, file_path, args.network, i)

        # model.eval()
        # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        # if args.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')


if __name__ == '__main__':
    args = parse_args()
    train_net(args)
