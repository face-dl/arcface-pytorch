# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function

import leveldb
import logging
import math
import os
import pickle
import time
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc


def load_bin(path, image_size):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = np.zeros((len(issame_list) * 2, image_size[0], image_size[1], 3))

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = cv2.imdecode(np.fromstring(bytes(_bin), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != image_size[0]:
            img = cv2.resize(img, image_size)
        data[i][:] = img
        if i % 1000 == 0:
            logging.info('loading bin %s', i)
    return (data, issame_list)


def get_featurs(model, images_lists, batch_size=10):
    features = None
    count = math.ceil(len(images_lists) / batch_size)
    for index in range(count):
        images = images_lists[index * batch_size:(index + 1) * batch_size, ...]
        data = torch.from_numpy(images)
        data = data.to(torch.device("cuda"))
        output = model(data)
        output = output.data.cpu().numpy()

        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))

        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))
            # logging.info("index/count %s/%s", index, count)
    logging.info("features shape %s", features.shape)
    return features


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(features, issame_list):
    sims = []
    labels = []
    for index, label in enumerate(issame_list):
        f = features[index]
        fe_1 = f[:512]
        fe_2 = f[512:]
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, path, batch_size):
    s = time.time()
    images, issame_list = load_bin(path, [112, 112])
    features = get_featurs(model, images, batch_size=batch_size)
    t = time.time() - s
    logging.info('total time is {}'.format(t))
    acc, th = test_performance(features, issame_list)
    logging.info('lfw face verification accuracy: %s th %s ', acc, th)
    return acc


class MaysaRoc(object):
    def __init__(self, leveldb_path, label_path, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        pic_db = leveldb.LevelDB(leveldb_path, max_open_files=100)
        self.images = []
        self.labels = []

        with open(label_path, "r") as file:
            label_probes = defaultdict(list)
            label_galleries = defaultdict(list)
            lines = file.readlines()
            for index, line in enumerate(lines):
                item = line.strip().split(",")
                pic_id, label, is_probe = item[0], item[1], item[2]
                label = int(label)
                self.labels.append(label)
                is_probe = int(is_probe)
                if is_probe:
                    label_probes[label].append(index)
                else:
                    label_galleries[label].append(index)
                try:
                    pic_id = str(pic_id).encode('utf-8')
                    data = pic_db.Get(pic_id)
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.images.append(img)
                except Exception as e:
                    logging.info("pic_id %s no pic", pic_id)

            search_paires = []
            for l in label_probes:
                probes = label_probes[l]
                galleries = label_galleries[l]
                for p in probes:
                    for g in galleries:
                        search_paires.append((p, g))
            self.search_paires = search_paires
        del pic_db
        logging.info("MaysaRoc load images %s search_paires %s", len(self.images), len(self.search_paires))
        self.labels = np.array(self.labels)
        self.images = np.array(self.images)

    def get_features(self, model):
        features = None
        batch_size = self.batch_size
        count = math.ceil(len(self.images) / batch_size)
        for index in range(count):
            images = self.images[index * batch_size:(index + 1) * batch_size, ...]
            feature = model(images)
            feature = feature / np.linalg.norm(feature, axis=1, keepdims=1)
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))
                # logging.info("index/count %s/%s", index, count)
        logging.info("get_features features shape %s", features.shape)
        return features

    def dis(self, vec1, vec2):
        similarity = np.dot(np.array(vec1), np.array(vec2).T)
        return similarity

    def roc(self, model, epoch):
        features = self.get_features(model)
        labels = self.labels

        scrub_labels = labels
        distractors_labels = labels
        results = self.dis(features, features)

        # cal search
        search_paires_simes = np.zeros(len(self.search_paires), dtype=np.float32)
        for i, p in enumerate(self.search_paires):
            search_paires_simes[i] = results[p[0], p[1]]
        x_labels = []
        y_labels = []
        for i in range(100):
            th = i * 0.01
            x_labels.append(th)
            y_labels.append(np.sum(search_paires_simes > th))
        fig = plt.figure()
        plt.plot(x_labels, y_labels, label=('th/count 0.40/%0.4f' % y_labels[40]))
        plt.xlim([0, 1.0])
        plt.grid(linestyle='--', linewidth=1)
        plt.xlabel('th')
        plt.ylabel('search')
        plt.title('Maysa rank')
        plt.legend(loc="lower right")
        fig.savefig(os.path.join(self.file_path, "rank_{}.jpg".format(epoch)))

        # cal roc
        roc_label = []
        roc_score = []
        for i, distractor_items in enumerate(results):
            for j, d in enumerate(distractor_items):
                # if scrub_labels[i] == distractors_labels[j]:
                if i < j:
                    continue
                if scrub_labels[i] == distractors_labels[j] and i != j:
                    roc_label.append(1)
                    roc_score.append(d)
                if scrub_labels[i] != distractors_labels[j]:
                    roc_label.append(0)
                    roc_score.append(d)
        logging.info("pos %s neg %s", sum(roc_label), len(roc_label) - sum(roc_label))
        x_labels = []
        for i in range(-10, 1):
            x_labels.append(10 ** i)
        fig = plt.figure()
        fpr, tpr, thresholds = roc_curve(roc_label, roc_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=('AUC = %0.4f' % roc_auc))
        plt.xlim(x_labels[0], x_labels[-1])
        plt.ylim([0, 1.0])
        plt.grid(linestyle='--', linewidth=1)
        plt.xticks(x_labels)
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.xscale('log')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Maysa ROC for labels 15 images 1317')
        plt.legend(loc="lower right")
        logging.info("head is fpr, content is tpr/threshold")
        fig.savefig(os.path.join(self.file_path, "roc_{}.jpg".format(epoch)))

        # tpr_fpr_table = PrettyTable(['Methods'] + x_labels)
        # tpr_fpr_row = []
        # tpr_fpr_row.append("baseline")
        # for fpr_iter in np.arange(len(x_labels)):
        #     tmp = [i + 100 if i < 0 else i for i in fpr - x_labels[fpr_iter]]
        #     _, min_index = min(list(zip(tmp, range(len(fpr)))))
        #     tpr_fpr_row.append('%.4f/%f' % (tpr[min_index], thresholds[min_index]))
        # tpr_fpr_table.add_row(tpr_fpr_row)
        # logging.info(tpr_fpr_table)
        return y_labels[40], roc_auc


def torch_model(model_name):
    from models import fmobilefacenet
    from torch.nn import DataParallel

    model = fmobilefacenet.resnet_face18(512)
    # model_name = "noise_v26_2020-01-07-00:29:38"
    # model_name = "noise_v26_2020-01-08-00:43:31,6"
    # model_name = "noise_v41_2020-01-12-23:13:57,8"
    # model_name = "noise_v42_2020-01-11-21:33:18,8"
    # model_name = "noise_v43_2020-01-10-15:21:51,7"
    # model_name = "noise_v44_2020-01-08-19:07:06,8"

    # pretrained = os.path.expanduser('./train/noise_2020-01-04-23:17:15/resnet18,2')
    # pretrained = os.path.expanduser('./train/noise_2020-01-05-23:39:39_back/resnet18,16')

    model_name, epoch = model_name.split(",")
    pretrained = os.path.expanduser('./train/{}/resnet18,6'.format(model_name, epoch))
    pretrained, iter_cnt = pretrained.split(",")

    model = DataParallel(model)
    model.load_state_dict(torch.load(pretrained + '_base_' + str(iter_cnt) + '.pth'))
    model.to(torch.device("cuda"))
    model.eval()

    def feature_func(images):
        data = torch.from_numpy(images)
        data = data.to(torch.device("cuda"))
        feat = model(data)
        feat = feat.data.cpu().numpy()
        return feat

    return model_name, feature_func


def mx_model():
    import mxnet as mx
    model_name = "model-官方retina"
    model_name = "model-v16-0.00001"
    model_name = "model,0"
    # model_name = "model-v28-0.0001"
    # # model_name = "model-v28-0.00001"
    # model_name = "model-v27-0.00001"
    # model_name = "model-v26-0.00001"
    # model_name = "model-v19-0.00001"
    # model_name = "model-v18-0.00001"
    # model_name = "model-v17-0.00001"
    # model_name = "model-v23"
    # model_name = "model-v25-0.00001"
    # model_name = "model-v22-0.00001"
    # model_name = "model-v24-0.00001"
    # model_name = "model-v20-0.00001"
    # model_name = "model-v22-0.0001"
    # model_name = "model-v20-0.0001"
    # model_name = "model-v16-0.0001"
    # model_name = "model-v15-11-0.00001"
    model_name = "model-v15-9-0.0001,343490"
    # model_name = "model-v15-base-0.00001"
    # model_name = "model-10wan-高斯处理"
    # model_name = "model-maysa高斯处理0.00002"
    # model_name = "model-maysa高斯处理0.002"
    # model_name = "model-maysa训练的模型"
    # model_name = "model-线上版本"
    pretrained = os.path.expanduser('/opt/face/models/insight/v14/{}'.format(model_name))
    prefix, epoch = pretrained.split(",")
    ctx = mx.gpu()

    def load_checkpoint(prefix, epoch):
        symbol = mx.sym.load('/opt/face/models/insight/v14/model-symbol.json')
        save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        return (symbol, arg_params, aux_params)

    sym, arg_params, aux_params = load_checkpoint(prefix, int(epoch))
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    image_size = (112, 112)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    def feature_func(data):
        data = mx.nd.array(data, ctx=ctx)
        data = data.transpose((0, 3, 1, 2))
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        feat = model.get_outputs()[0].asnumpy()
        return feat

    return model_name, feature_func


def merge_all_label():
    test_labels = []
    test_labels.append(os.path.expanduser("~/datasets/cacher/xm_bailujun.labels"))
    test_labels.append(os.path.expanduser("/opt/face/caches/labels/0371d825c3.labels"))
    test_labels.append(os.path.expanduser("/opt/face/caches/labels/xm_jinyutixiang.labels"))
    test_labels.append(os.path.expanduser("/opt/face/caches/labels/xm_chengshizhiguang.labels"))
    test_labels.append(os.path.expanduser("/opt/face/caches/labels/xm_lucheng.labels"))
    test_labels.append(os.path.expanduser("/opt/face/caches/labels/xm_jiulongtai.labels"))

    all_target = os.path.expanduser("/opt/face/caches/labels/all.labels")

    with open(all_target, "w") as target:
        for dataset_index, label_path in enumerate(test_labels):
            with open(label_path, "r") as file:
                lines = file.readlines()
                for index, line in enumerate(lines):
                    item = line.strip().split(",")
                    pic_id, label, is_probe = item[0], item[1], item[2]
                    label = int(label)
                    label += dataset_index * 10000
                    target.write("{},{},{}\n".format(pic_id, label, is_probe))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # model_name, feature_func = torch_model()
    # model_name, feature_func = mx_model()

    # target = os.path.expanduser("~/datasets/maysa/lfw.bin")
    # lfw_test(model, target, 64)

    leveldb_path = os.path.expanduser("~/datasets/cacher/pictures")
    test_labels = os.path.expanduser("~/datasets/cacher/xm_bailujun.labels")
    test_labels = os.path.expanduser("/opt/face/caches/labels/0371d825c3.labels")
    test_labels = os.path.expanduser("/opt/face/caches/labels/xm_jinyutixiang.labels")
    test_labels = os.path.expanduser("/opt/face/caches/labels/xm_chengshizhiguang.labels")
    test_labels = os.path.expanduser("/opt/face/caches/labels/xm_lucheng.labels")
    test_labels = os.path.expanduser("/opt/face/caches/labels/xm_jiulongtai.labels")

    # merge_all_label()
    test_labels = os.path.expanduser("/opt/face/caches/labels/all.labels")

    project = test_labels.split("/")[-1].split(".")[0]
    roc_path = "roc_{}".format(project)
    if not os.path.exists(roc_path):
        os.mkdir(roc_path)

    datas = []
    for model_config in ["noise_v26_2020-01-08-00:43:31,6", "noise_v40_2020-01-14-18:06:48,8", "noise_v41_2020-01-12-23:13:57,8", "noise_v42_2020-01-11-21:33:18,8",
                         "noise_v43_2020-01-10-15:21:51,8", "noise_v44_2020-01-08-19:07:06,8"]:
        model_name, feature_func = torch_model(model_config)
        maysa_roc = MaysaRoc(leveldb_path=leveldb_path, label_path=test_labels, file_path=roc_path, batch_size=64)
        count, roc_auc = maysa_roc.roc(feature_func, model_name)
        datas.append([count, roc_auc])

    logging.info("datas %s", datas)
