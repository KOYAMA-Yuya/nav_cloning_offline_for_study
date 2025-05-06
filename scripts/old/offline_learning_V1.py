#!/usr/bin/env python3
import os
import sys
import csv
import time
import cv2
import roslib
import numpy as np
from nav_cloning_pytorch import deep_learning
from skimage.transform import resize

class CourseFollowingLearningNode:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.model_num = str(sys.argv[1])
        self.pro = "20250430_22:22:12"  # データセットの識別名

        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/'
        self.save_path = self.path + f"model/{self.pro}/model{self.model_num}.pt"
        self.ang_path = self.path + f"ang/{self.pro}/"
        self.img_path = self.path + f"img/{self.pro}/"
        
        self.learn_no = 4000
        self.data =  1704 # 使用するデータ数
        self.BATCH_SIZE = 32
    
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs(self.path + f"/loss/{self.pro}/", exist_ok=True)

    def load_images(self, index, parallax):
        
        img_types = ["center"]  #中央画像のみ
        #img_types = ["left", "center", "right"]
        images = []

        for img_type in img_types:
            
            # 画像ファイルのパスを作成
            img_file = f"{self.img_path}{img_type}{index}_{parallax}.jpg"
            img = cv2.imread(img_file)
            #リサイズ　あとでsetcollectの方にしたい
            img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
            
            
            if img is None:
                print(f"Warning: Failed to load {img_file}")
                continue
            
            images.append(img)

        return images


    def load_angles(self):
        angles = []
        with open(self.ang_path + 'ang.csv', 'r') as f:
            for row in csv.reader(f):
                _, tar_ang = row
                angles.append(float(tar_ang))
        return angles

    def learn(self):
        ang_list = self.load_angles()
        ang_number = 0

        for i in range(self.data):
            for parallax in ["-5", "0", "+5"]:
                images = self.load_images(i,parallax)
                ang_number += 1
                target_ang = ang_list[ang_number]
                self.dl.make_dataset(images, target_ang)
            
            print(f"Dataset: {i}, target_ang:{target_ang}")
        
        loss_log = []
        for l in range(self.learn_no):
            start_time_epoch = time.time()
            loss = self.dl.trains(self.BATCH_SIZE)
            end_time_epoch = time.time()
            print(f"Epoch {l + 1}, Loss: {loss}, Epoch time: {end_time_epoch - start_time_epoch:.4f} seconds")
            loss_log.append([str(loss)])
            
        self.loss_path = os.path.join(self.path, f"loss/{self.pro}/{self.model_num}.csv")
        with open(self.loss_path, 'a') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            writer.writerows(loss_log)
        
        self.dl.save(self.save_path)
        print(self.path, f"loss/{self.pro}/{self.model_num}.csvに保存しました")
        sys.exit()

if __name__ == '__main__':
    node = CourseFollowingLearningNode()
    node.learn()
