## **コードの全体構成**

#### 1. **インポート**
```python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *
from skimage.transform import resize
from geometry_msgs.msg import Twist, PoseArray, PoseWithCovarianceStamped
from std_msgs.msg import Int8
from std_srvs.srv import SetBool, SetBoolResponse, Trigger
from nav_msgs.msg import Path, Odometry
from gazebo_msgs.msg import ModelStates
import csv
import os
import time
import sys
import copy
import tf
```
- 必要なROSのメッセージ型やサービス、外部ライブラリ（OpenCVや深層学習用のネットワークなど）をインポート。

#### 2. **`nav_cloning_node` クラス**
```python
class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        # 初期化とROSノードの作成
```
- ROSノードを初期化し、各種パラメータやサブスクライバ、パブリッシャを設定。

#### 3. **各種サブスクライバの設定**
```python
self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
```
- カメラ画像や速度、ロボットの位置（AMCL）、経路などのデータを受信するサブスクライバを設定。

#### 4. **ディープラーニングモデルの設定**
```python
self.dl = deep_learning(n_action=self.action_num)
```
- `deep_learning` クラス（`nav_cloning_net` モジュール内に定義されていると思われる）を使って、ロボットの行動制御を学習。 `n_action` はロボットの可能なアクション数を指定。

#### 5. **コールバック関数**
- **`callback`**: カメラ画像を受け取って `cv_image` に保存。
- **`callback_vel`**: 受け取った速度情報を元にロボットのアクションを決定。
- **`callback_pose`**: ロボットの現在位置を受け取り、目的地までの最短距離を計算。
- **`callback_dl_training`**: トレーニングの開始・停止を切り替え。

#### 6. **画像処理と学習ループ**
```python
def loop(self):
    if self.cv_image.size != 640 * 480 * 3:
        return
    img = resize(self.cv_image, (48, 64), mode='constant')
    img_left = resize(self.cv_left_image, (48, 64), mode='constant')
    img_right = resize(self.cv_right_image, (48, 64), mode='constant')
```
- カメラ画像をリサイズしてニューラルネットワークに渡しやすい形式に変換。

#### 7. **モードごとの挙動**
- **`manual`**: 手動モード。ロボットの位置が一定の距離に近づいたときに深層学習を使うか決定。
- **`zigzag`**: ジグザグモード。深層学習を使ってアクションを決定。
- **`use_dl_output`**: 深層学習による制御をそのまま使用するモード。
- **`follow_line`**: ラインに沿って進む動作を学習。
- **`selected_training`**: 特定の学習条件でトレーニングを行う。

```python
if self.learning:
    action, loss = self.dl.act_and_trains(img, target_action)
    # 深層学習によるアクションを選択してトレーニング
    self.vel.linear.x = 0.2
    self.vel.angular.z = target_action
    self.nav_pub.publish(self.vel)
```
- 学習モードの場合、深層学習ネットワークを使って行動を決定し、ロボットに移動命令を出す。

#### 8. **テストモード**
```python
else:
    target_action = self.dl.act(img)
    # テストモードでは学習済みモデルを使って行動を決定
```
- 学習が終わった後、テストモードでは学習済みのネットワークを使ってロボットを制御。

#### 9. **画像の表示**
```python
cv2.imshow("Resized Image", temp)
cv2.waitKey(1)
```
- 処理した画像を表示。

#### 10. **CSVによるログ保存**
- ロボットの移動の過程やエラー情報などをCSVファイルに記録。
```python
with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(line)
```