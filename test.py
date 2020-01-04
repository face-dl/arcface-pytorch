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
from torch.nn import DataParallel


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
    logging.info("features shape %s", feature.shape)
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
    print(features.shape)
    t = time.time() - s
    logging.info('total time is {}'.format(t))
    acc, th = test_performance(features, issame_list)
    logging.info('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
