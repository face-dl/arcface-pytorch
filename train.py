from __future__ import print_function

import argparse
import logging
from collections import defaultdict

import git
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter

from data.dataset import Dataset
from models import resnet, metrics, fmobilefacenet
from models.focal_loss import FocalLoss
from test import *
from utils.visualizer import Visualizer


class Noise(object):
    def __init__(self, file_path, poche_size):
        self.consine_list = []
        self.file_path = file_path
        self.poche_size = poche_size
        self.sample_len = 5000
        self.save_freq = 1000

    def save_png(self, cos_t_cur, end_str):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        ###绘图
        plt.hist(cos_t_cur, bins=1000, color='g')
        plt.title('余弦直方图')
        ###保存
        plt.savefig("{}/plt_{}.jpg".format(self.file_path, end_str))

    def save_mean_png(self, cos_t_cur, end_str):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        ###绘图
        x = np.arange(len(cos_t_cur))
        plt.plot(x, cos_t_cur, color="red", linewidth=1)
        plt.title('余弦柱状图')
        ###保存
        plt.savefig("{}/plt_mean_{}.jpg".format(self.file_path, end_str))

    def cal_thes(self, n2, end_str):
        sigm_l_th = int(len(n2) * 0.005)
        sigm_l = n2[sigm_l_th]
        sigm_r = n2[-sigm_l_th]
        logging.info("sigm_l %s sigm_r %s", sigm_l, sigm_r)

        bins = 100
        bin_dict = defaultdict(int)
        for n in n2:
            n = int(n * bins)
            if n == bins:
                n == bins - 1
            bin_dict[n] += 1

        n2 = np.zeros(bins * 2)
        for index in range(bins * 2):
            n2[index] = bin_dict[index - bins]
        n2_copy = n2.copy()
        for index in range(bins * 2):
            start = max(0, index - 2)
            end = min(bins * 2, index + 2)
            n2[index] = np.mean(n2_copy[start:end])

        self.save_mean_png(n2, end_str)
        max_thes = []
        max_counts = []
        for th in range(bins * 2):
            if 5 <= th <= bins * 2 - 6:
                is_max = True
                for i in range(11):
                    cur = n2[th - 5 + i]
                    if cur == 0 or cur > n2[th]:
                        is_max = False
                        break
                if is_max:
                    max_thes.append((th - bins) * 0.01)
                    max_counts.append(n2[th])
        logging.info("max_thes %s max_counts %s", max_thes, max_counts)

    def append_cosine(self, consine, nbatch):
        self.consine_list.append(consine)
        if nbatch % self.save_freq == 0:
            cos_t_cur = np.concatenate(self.consine_list[-self.sample_len:])
            self.save_png(cos_t_cur, nbatch)
            cos_t_cur.sort()
            self.cal_thes(cos_t_cur, nbatch)

    def save_epoch(self, epoch):
        cos_t_all = np.concatenate(self.consine_list[-self.poche_size:])
        self.save_png(cos_t_all, "epoch_{}".format(epoch))
        cos_t_all.sort()
        self.cal_thes(cos_t_all, "epoch_{}".format(epoch))


class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self, name):
        self.name = str(name)
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_value(self):
        return self.sum_metric / self.num_inst

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))


class LossMetric(EvalMetric):
    def __init__(self):
        super(LossMetric, self).__init__('loss')

    def update_loss(self, loss):
        self.num_inst += 1
        self.sum_metric += loss


class ThetaMetric(EvalMetric):
    def __init__(self):
        super(ThetaMetric, self).__init__("theta")

    def update(self, labels, cosine):
        indexes = torch.LongTensor(labels).unsqueeze(dim=1)
        consine_list = cosine.gather(1, indexes)
        self.num_inst += 1
        mean_rad = consine_list.acos().mean().item()
        self.sum_metric += np.rad2deg(mean_rad)


class AccMetric(EvalMetric):
    def __init__(self, is_real):
        super(AccMetric, self).__init__("real_acc" if is_real else "acc")

    def update(self, labels, cosine):
        cosine = np.argmax(cosine, axis=1)
        acc = np.mean((cosine == labels).astype(int))
        self.num_inst += 1
        self.sum_metric += acc


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

    parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in each context')
    parser.add_argument('--num_workers', type=int, default=2, help='batch size in each context')
    parser.add_argument('--loss', type=str, default="focal_loss", help='batch size in each context')
    parser.add_argument('--metric', type=str, default="arc_margin", help='batch size in each context')

    parser.add_argument('--pretrained', default='./train/noise_2020-01-04-19:56:40/resnet18,10', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='', help='pretrained model to load')

    parser.add_argument('--network', default='resnet18', help='specify network')
    parser.add_argument('--optimizer', default='sgd', help='specify network')
    parser.add_argument('--margin_s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin_m', type=float, default=0.5, help='margin for loss,')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')

    parser.add_argument('--lr_steps', type=str, default='1,5', help='steps of lr changing')
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
    sw = SummaryWriter(file_path)
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
    # logging.info(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    if args.pretrained:
        pretrained, iter_cnt = args.pretrained.split(",")
        model.load_state_dict(torch.load(pretrained + '_base_' + str(iter_cnt) + '.pth'))
        metric_fc.load_state_dict(torch.load(pretrained + '_weight_' + str(iter_cnt) + '.pth'))

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr, weight_decay=args.weight_decay)

    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    logging.info("lr_steps %s", lr_steps)
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    if len(lr_steps) == 1:
        max_epoch = lr_steps[0] * 2
    else:
        max_epoch = 2 * lr_steps[-1] - lr_steps[-2]
    start = time.time()
    loss_metric = LossMetric()
    theta_metric = ThetaMetric()
    acc_metric = AccMetric(False)
    real_acc_metric = AccMetric(True)
    noise = Noise(file_path, len(trainloader))
    for i in range(max_epoch):
        model.train()

        for ii, data in enumerate(trainloader):
            iters = i * len(trainloader) + ii

            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            cosine, output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add metrics
            label = label.data.cpu().numpy()
            loss_metric.update_loss(loss.item())
            cosine = cosine.data.cpu()
            theta_metric.update(label, cosine)
            acc_metric.update(label, output)
            real_acc_metric.update(label, cosine)
            noise.append_cosine(cosine, iters)

            if iters % args.print_freq == 0:
                mean_loss = loss_metric.get_value()
                mean_theta = theta_metric.get_value()
                acc = acc_metric.get_value()
                real_acc = real_acc_metric.get_value()

                cost = (time.time() - start) / 3600
                left = cost / (iters + 1) * (len(trainloader) * max_epoch - (iters + 1))
                time_str = time.asctime(time.localtime(time.time()))
                logging.info('time %s train lr %.02f epoch/max_epoch %s/%s iter/size %s/%s iters %s cost/left %.02f/%.02f loss %.02f mean_theta %.02f acc %.02f real_acc %.02f',
                             time_str, optimizer.param_groups[0]['lr'], i, max_epoch, ii, len(trainloader), iters, cost, left, mean_loss, mean_theta, acc, real_acc)

                if args.display:
                    visualizer.display_current_results(iters, mean_loss, name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')
                sw.add_scalar("loss", mean_loss, iters)
                sw.add_scalar("theta", mean_theta, iters)
                sw.add_scalar("acc", acc, iters)
                sw.add_scalar("real_acc", real_acc, iters)

        noise.save_epoch(i)
        save_model(model, metric_fc, file_path, args.network, i)
        scheduler.step()

        # model.eval()
        # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        # if args.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')


if __name__ == '__main__':
    args = parse_args()
    train_net(args)
