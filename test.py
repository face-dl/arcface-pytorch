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
import numpy as np
import torch


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


if __name__ == '__main__':
    from models import fmobilefacenet

    model = fmobilefacenet.resnet_face18(512)
    pretrained = os.path.expanduser('./train/noise_2020-01-04-23:17:15/resnet18,2')
    pretrained, iter_cnt = pretrained.split(",")
    model.load_state_dict(torch.load(pretrained + '_base_' + str(iter_cnt) + '.pth'))
    model.to(torch.device("cuda"))

    model.eval()
    target = os.path.expanduser("~/datasets/maysa/lfw.bin")
    lfw_test(model, target, 64)
