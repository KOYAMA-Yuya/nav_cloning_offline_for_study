#!/usr/bin/env python3
import os
import sys
import time
import csv
import cv2
from skimage.transform import resize
import random
from nav_cloning_pytorch import deep_learning

class CourseFollowingLearningNode:
    def __init__(self):
        # モデルの初期化
        self.model = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.model_num = str(sys.argv[1])  # 実行時の引数としてモデル番号を受け取る
        
        # データセット名（記録用の識別名）
        self.dataset_name = "20250404_19:25:01"  # 実際の走行データの名前に変更する
        
        # データ保存用パス
        base_path = "/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data"
        self.model_path = f"{base_path}/model/{self.dataset_name}/model{self.model_num}.pt"
        self.ang_path = f"{base_path}/ang/{self.dataset_name}/ang.csv"
        self.img_dirs = {
            "left": f"{base_path}/img/{self.dataset_name}/left",
            "center": f"{base_path}/img/{self.dataset_name}/center",
            "right": f"{base_path}/img/{self.dataset_name}/right"
        }
        self.loss_path = f"{base_path}/loss/{self.dataset_name}/{self.model_num}.csv"
        
        # データ数と学習回数
        self.num_data = 406  # 画像と舵角データのペア数
        self.num_epochs = 4000  # 学習回数
        
        # 保存先ディレクトリを作成
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.loss_path), exist_ok=True)
    
    def load_data(self):
        """
        画像データと舵角データを読み込む
        """
        images = []
        angles = []

        # 舵角データの読み込み
        with open(self.ang_path, 'r') as f:
            for row in csv.reader(f):
                _, angle = row
                angles.append(float(angle))

        # 画像データの読み込み
        for i in range(self.num_data):
            img_set = {}
            for cam in ["left", "center", "right"]:
                img_set[cam] = {
                    offset: cv2.imread(f"{self.img_dirs[cam]}/{i}_{offset}.jpg")
                    for offset in ["+5", "0", "-5"]
                }
            images.append(img_set)

        return images, angles
    
    def prepare_dataset(self, images, angles):
        """
        読み込んだデータを学習データとしてモデルに登録
        """
        angle_offsets = {
            "left": {"+5": -0.244, "0": -0.182, "-5": -0.057},
            "center": {"+5": -0.0128, "0": 0, "-5": 0.134},
            "right": {"+5": 0.196, "0": 0.245, "-5": 0.26}
        }

        for i, (img_set, angle) in enumerate(zip(images, angles)):
            for cam in ["left", "center", "right"]:
                for offset, img in img_set[cam].items():
                    if img is not None:
                        self.model.make_dataset(img, angle + angle_offsets[cam][offset])
            print(f"Dataset added: {i + 1}/{self.num_data}")
    
    def train_model(self):
        """
        モデルの学習を実行
        """
        with open(self.loss_path, 'a') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            for epoch in range(self.num_epochs):
                loss = self.model.trains(self.num_data)
                writer.writerow([str(loss)])
                print(f"Training Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss}")
    
    def save_model(self):
        """
        学習済みモデルを保存
        """
        self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}")

    def run(self):
        """
        学習プロセスを実行
        """
        images, angles = self.load_data()
        self.prepare_dataset(images, angles)
        self.train_model()
        self.save_model()
        sys.exit()

if __name__ == '__main__':
    node = CourseFollowingLearningNode()
    node.run()
