#!/usr/bin/env python3
import os
import sys
import csv
import time
import cv2
import random
import numpy as np
from nav_cloning_pytorch import deep_learning

class CourseFollowingLearningNode:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.model_num = str(sys.argv[1])
        self.pro = "20250419_17:10:14"  # データセットの識別名
        self.save_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/model/{self.pro}/model{self.model_num}.pt"
        self.ang_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/ang/{self.pro}/"
        self.img_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/img/{self.pro}/"
        self.data = 616  # 使用するデータ数
        self.BATCH_SIZE = 32  # バッチサイズを指定
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs(f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/loss/{self.pro}/", exist_ok=True)

    def load_images(self, index):
        shifts = {
            "left": -0.2,
            "center": 0,
            "right": 0.2
        }

        img_types = ["left", "center", "right"]
        images = []

        for img_type in img_types:
            angle_shift = shifts[img_type]
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

    def learn(self, EPOCHS=50):
        ang_list = self.load_angles()

        # --- データのインデックスをシャッフル ---
        indices = list(range(self.data))
        random.shuffle(indices)

        # データセット作成（シャッフル後）
        for i in indices:
            images = self.load_images(i)
            target_ang = ang_list[i]
            for img, angle_shift in images:
                self.dl.make_dataset(img, target_ang + angle_shift)
            print(f"Dataset: {i}")

        loss_log = []

        # 学習処理：エポックごとにループ
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            
            # バッチ処理
            for i in range(0, len(self.dl.x_cat), self.BATCH_SIZE):
                # 現在のバッチ
                batch_x = self.dl.x_cat[i:i + self.BATCH_SIZE]
                batch_t = self.dl.t_cat[i:i + self.BATCH_SIZE]
                
                loss = self.dl.trains(self.BATCH_SIZE)
                loss_log.append([str(loss)])
                print(f"Epoch {epoch + 1}, Batch {i//self.BATCH_SIZE + 1}, Loss: {loss}")

        # ロスの保存
        loss_path = f"/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/loss/{self.pro}/{self.model_num}.csv"
        with open(loss_path, 'a') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            writer.writerows(loss_log)

        # モデルの保存
        self.dl.save(self.save_path)
        print(f"モデル保存完了: {self.save_path}")

        sys.exit()

if __name__ == '__main__':
    node = CourseFollowingLearningNode()
    node.learn()