# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function

import logging
import math
import os
import pickle
import time

import cv2
import leveldb
import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
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
            lines = file.readlines()
            for index, line in enumerate(lines):
                item = line.strip().split(",")
                pic_id, label = item[0], item[1]
                label = int(label)
                self.labels.append(label)
                try:
                    pic_id = str(pic_id).encode('utf-8')
                    data = pic_db.Get(pic_id)
                    img = cv2.imdecode(np.fromstring(bytes(data), np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.images.append(img)
                except Exception as e:
                    logging.info("pic_id %s no pic", pic_id)
        del pic_db
        logging.info("MaysaRoc load images %s", len(self.images))
        self.labels = np.array(self.labels)
        self.images = np.array(self.images)

    def get_features(self, model):
        features = None
        batch_size = self.batch_size
        count = math.ceil(len(self.images) / batch_size)
        for index in range(count):
            images = self.images[index * batch_size:(index + 1) * batch_size, ...]
            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            feature = output.data.cpu().numpy()
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
        for i in range(-5, 1):
            x_labels.append(10 ** i)
        tpr_fpr_table = PrettyTable(['Methods'] + x_labels)
        fig = plt.figure()

        fpr, tpr, thresholds = roc_curve(roc_label, roc_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=('AUC = %0.4f' % roc_auc))
        tpr_fpr_row = []
        tpr_fpr_row.append("baseline")
        for fpr_iter in np.arange(len(x_labels)):
            tmp = [i + 100 if i < 0 else i for i in fpr - x_labels[fpr_iter]]
            _, min_index = min(list(zip(tmp, range(len(fpr)))))
            tpr_fpr_row.append('%.4f/%f' % (tpr[min_index], thresholds[min_index]))
        tpr_fpr_table.add_row(tpr_fpr_row)

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
        logging.info(tpr_fpr_table)
        fig.savefig(os.path.join(self.file_path, "roc_{}.jpg".format(epoch)))


if __name__ == '__main__':
    from models import fmobilefacenet
    from torch.nn import DataParallel

    model = fmobilefacenet.resnet_face18(512)
    pretrained = os.path.expanduser('./train/noise_2020-01-04-23:17:15/resnet18,2')
    pretrained, iter_cnt = pretrained.split(",")

    model = DataParallel(model)
    model.load_state_dict(torch.load(pretrained + '_base_' + str(iter_cnt) + '.pth'))
    model.to(torch.device("cuda"))

    model.eval()
    target = os.path.expanduser("~/datasets/maysa/lfw.bin")
    lfw_test(model, target, 64)
