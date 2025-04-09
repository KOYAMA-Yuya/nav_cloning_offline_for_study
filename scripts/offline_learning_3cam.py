#!/usr/bin/env python3
import os
import sys
import csv
import time
import cv2
import numpy as np
from nav_cloning_pytorch import deep_learning

class CourseFollowingLearningNode:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.model_num = str(sys.argv[1])
        self.pro = "20250409_00:14:17"  # データセットの識別名
        self.save_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/model/{self.pro}/model{self.model_num}.pt"
        self.ang_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/ang/{self.pro}/"
        self.img_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/img/{self.pro}/"
        self.learn_no = 2000
        self.data = 1080  # 使用するデータ数
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs(f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/loss/{self.pro}/", exist_ok=True)

    def load_images(self, index):
        # 各カメラタイプに対応する角度シフトの設定
        shifts = {
            "left": -0.17,
            "center": 0,
            "right": 0.17
        }

        img_types = ["left", "center", "right"]
        images = []

        for img_type in img_types:
            # img_typeに対応する角度シフトを取得
            angle_shift = shifts[img_type]
            
            # 画像ファイルのパスを作成
            img_file = f"{self.img_path}{img_type}{index}.jpg"
            img = cv2.imread(img_file)
            
            if img is None:
                print(f"Warning: Failed to load {img_file}")
                continue

            images.append((img, angle_shift))

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
        
        for i in range(self.data):
            images = self.load_images(i)
            target_ang = ang_list[i]
            for img, angle_shift in images:
                self.dl.make_dataset(img, target_ang + angle_shift)
            print(f"Dataset: {i}")
        
        loss_log = []
        for l in range(self.learn_no):
            loss = self.dl.trains(self.data)
            loss_log.append([str(loss)])
            print(f"Train step: {l}, Loss: {loss}")
        
        loss_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/loss/{self.pro}/{self.model_num}.csv"
        with open(loss_path, 'a') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            writer.writerows(loss_log)
        
        self.dl.save(self.save_path)
        print(f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/model/{self.pro}/model{self.model_num}.csvに保存しました")
        sys.exit()

if __name__ == '__main__':
    node = CourseFollowingLearningNode()
    node.learn()
