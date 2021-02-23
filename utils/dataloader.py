# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
from p2m.config import *
import numpy as np
import pickle
import scipy.sparse as sp
import networkx as nx
import threading
import queue
import sys
import cv2
import math
import time
import os
import glob
import pprint


np.random.seed(1)


class DataFetcher(threading.Thread):
    '''
    filelist:data/train_list.txt
    dataroot:/workspace/tf2_gcn-main/data/ShapeNetModels/p2mppdata/train
    imageroot:/workspace/tf2_gcn-main/data/ShapeNetImages/ShapeNetRendering
    '''
    def __init__(self, file_list, data_root, image_root, is_val=False, mesh_root=None):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = queue.Queue(64)
        self.data_root = data_root
        self.image_root = image_root
        self.is_val = is_val

        self.pkl_list = []
        with open(file_list, 'r') as f:
            while True:
                line = f.readline().strip()#strip 去除首尾字符空格
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.mesh_root = mesh_root
        self.number = len(self.pkl_list)#number=35010
       
        np.random.shuffle(self.pkl_list)
    def run(self):
        #被执行
        while self.index < 9000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number)) #%取余
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)
    def work(self, idx):
        ''''''
        pkl_item = self.pkl_list[idx]
        pkl_path = os.path.join(self.data_root, pkl_item)
        pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
        if self.is_val:
            label = pkl[1]
        else:
            label = pkl
        # load image file
        img_root = self.image_root
        ids = pkl_item.split('_')
        category = ids[-3]
        item_id = ids[-2]
        img_path = os.path.join(img_root, category, item_id,'rendering')
        camera_meta_data = np.loadtxt(os.path.join(img_path, 'rendering_metadata.txt'))
        if self.mesh_root is not None:
            mesh = np.loadtxt(os.path.join(self.mesh_root, category + '_' + item_id + '_00_predict.xyz'))
        else:
            mesh = None
        imgs = np.zeros((3, 224, 224, 3))
        poses = np.zeros((3, 5))
        for idx, view in enumerate([0, 6, 7]):
            img = cv2.imread(os.path.join(img_path, str(view).zfill(2) + '.png'), cv2.IMREAD_UNCHANGED)
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float32') / 255.0
            imgs[idx] = img_inp[:, :, :3]
            poses[idx] = camera_meta_data[view]
        return imgs, label, poses, pkl_item, mesh



    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()


if __name__ == '__main__':
    print('=> set config')
    yaml_path = '/workspace/3D/tf2_gcn-main/cfgs/mvp2m.yaml'
    cfg = execute(yaml_path)# 配置问题成功解决
    pprint.pprint(vars(cfg))
    data = DataFetcher(file_list=cfg.train_file_path, data_root=cfg.train_data_path, image_root=cfg.train_image_path, is_val=False)
    data.setDaemon(True)
    data.start()
    img_all_view, labels, poses, data_id, mesh = data.fetch()
    data.stopped = True
#/workspace/3D/tf2_gcn-main/data/ShapeNetModels/p2mppdata/train