import logging
import os
import random
from collections import defaultdict

import cv2
import leveldb
import numpy as np
import torch
import torchvision
from torch.utils import data as torch_data

logger = logging.getLogger()


class Dataset(torch_data.Dataset):
    pic_db_dict = {}

    def __init__(self, leveldb_path, label_path, min_images=0, max_images=11111111111, ignore_labels=set()):
        super(Dataset, self).__init__()
        assert leveldb_path
        logger.info('loading FaceDataset %s %s min_images %s max_images %s', leveldb_path, label_path, min_images, max_images)
        self.leveldb_path = leveldb_path
        self.label_path = label_path

        self.ignore_pic_ids = set()
        if leveldb_path in self.pic_db_dict:
            self.pic_db = self.pic_db_dict[leveldb_path]
        else:
            self.pic_db = leveldb.LevelDB(leveldb_path, max_open_files=100)
            self.pic_db_dict[leveldb_path] = self.pic_db
        with open(self.label_path, "r") as file:
            lines = file.readlines()
            self.base_pic_ids = []
            self.base_labels = []
            self.base_label2pic = defaultdict(list)
            for index, line in enumerate(lines):
                pic_id, label = line.strip().split(",")
                label = int(label)
                if label == -1 or label in ignore_labels:
                    continue
                self.base_pic_ids.append(pic_id)
                self.base_labels.append(label)
                self.base_label2pic[label].append(pic_id)
        self.min_images = min_images
        self.max_images = max_images
        self.reset()

    def reset(self):
        logger.info("origin pic_ids %s labels %s", len(self.base_pic_ids), len(self.base_label2pic))
        min_images = self.min_images
        max_images = self.max_images
        if min_images > 0:
            new_label2pic = defaultdict(list)
            for l in self.base_label2pic:
                c = self.base_label2pic[l]
                if len(c) >= min_images:
                    sub = random.sample(c, min(max_images, len(c)))
                    new_label2pic[l] = sub
            new_pic_ids = []
            new_labels = []
            for label in new_label2pic:
                if label in new_label2pic:
                    new_pic_ids += new_label2pic[label]
                    new_labels += [label] * len(new_label2pic[label])
            self.pic_ids = new_pic_ids
            self.labels = new_labels
            self.label2pic = new_label2pic
        else:
            self.pic_ids = self.base_pic_ids
            self.labels = self.base_labels
            self.label2pic = self.base_label2pic

        self.order_labels = sorted(self.label2pic.keys())
        self.train_labels = {}
        for index, label in enumerate(self.order_labels):
            self.train_labels[label] = index
        logger.info("final pic_ids %s labels %s", len(self.pic_ids), len(self.label2pic))

    @property
    def label_len(self):
        return len(self.label2pic)

    def __len__(self):
        return len(self.pic_ids)

    def __getitem__(self, idx):
        if idx < len(self.pic_ids):
            pic_id = self.pic_ids[idx]
            label = self.labels[idx]
            try:
                pic_id = str(pic_id).encode('utf-8')
                data = self.pic_db.Get(pic_id)
                img = cv2.imdecode(np.fromstring(bytes(data), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(img), self.train_labels[label]
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.info("pic_id %s no pic", pic_id)
                return torch.zeros((112, 112, 3)), -1
        else:
            print("get_item error")
            assert False


if __name__ == '__main__':
    leveldb_path = os.path.expanduser("/opt/cacher/faces_webface_112x112")
    label_path = os.path.expanduser("/opt/cacher/faces_webface_112x112.labels")
    dataset = Dataset(leveldb_path=leveldb_path, label_path=label_path)

    trainloader = torch_data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data.permute(0, 3, 1, 2)).numpy()
        # # print img.shape
        # # print label.shape
        # # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # # img *= np.array([0.229, 0.224, 0.225])
        # # img += np.array([0.485, 0.456, 0.406])
        # img += np.array([1, 1, 1])
        # img *= 127.5
        # img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        print(img.shape)
        cv2.imshow('img', img)
        cv2.waitKey()
        break
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
