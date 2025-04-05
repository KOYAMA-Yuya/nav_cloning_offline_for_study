#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def move_turtlebot():
    # ROSノードの初期化
    rospy.init_node('simple_cmd_vel', anonymous=True)

    # /cmd_vel トピックにメッセージを送信するためのパブリッシャーを作成
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # メッセージの作成
    move_cmd = Twist()

    # 定められた速度
    move_cmd.linear.x = 0.5  # 前進速度 (m/s)
    move_cmd.angular.z = 0.7  # 回転速度 (rad/s)

    # 1Hzでコマンドを送信
    rate = rospy.Rate(1)  # 1Hzでループ
    while not rospy.is_shutdown():
        cmd_vel_pub.publish(move_cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_turtlebot()
    except rospy.ROSInterruptException:
        pass



# #!/usr/bin/env python3

# import rospy
# import torch
# import cv2
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from geometry_msgs.msg import Twist

# from nav_cloning_pytorch import deep_learning

# class ImageToCmdVelNode:
#     def __init__(self):
#         rospy.init_node('image_to_cmd_vel', anonymous=True)
        
#         # 設定（元のコードに合わせる）
#         #self.num = rospy.get_param("/nav_cloning_node/num", "1")
#         self.pro = "20250404_22:09:55"
#         self.load_path = "/home/koyama-yuya/ros_ws/nav_cloning_offline_for_study_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model1.pt" #+str(self.num)+".pt"

#         # deep_learningのインスタンス化
#         self.dl = deep_learning(n_action = 1)  # ここでアクション数を指定（必要に応じて変更）

#         # モデルのロード
#         print("Loading model from:", self.load_path)
#         self.dl.load(self.load_path)  # deep_learningクラスにloadメソッドがあると仮定
        
#         # その他の設定（変数の初期化など）
#         self.bridge = CvBridge()
#         self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
#         self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

#         # 固定の前進速度 (m/s)
#         self.linear_speed = 1.0
        
#     def image_callback(self, msg):
#         try:
#             # ROSイメージメッセージをOpenCVフォーマットに変換
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
#             # 画像前処理（リサイズ & テンソル変換）
#             input_tensor = self.preprocess_image(cv_image)

#             # 学習済みモデルでハンドリング（舵角）を推論
#             steering_angle = self.predict_steering(input_tensor)

#             # 速度コマンドをパブリッシュ
#             self.publish_cmd_vel(steering_angle)

#         except Exception as e:
#             rospy.logerr("Image processing error: %s", str(e))

#     def preprocess_image(self, cv_image):
#         """画像をモデルに適した形式に前処理"""
#         img_resized = cv2.resize(cv_image, (224, 224))  # モデルの入力サイズに合わせる
#         img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)  # HWC → CHW
#         img_tensor = img_tensor.unsqueeze(0) / 255.0  # バッチ次元追加 & 正規化
#         return img_tensor

#     def predict_steering(self, input_tensor):
#         """学習済みモデルを使って舵角を予測"""
#         with torch.no_grad():
#             output = self.dl.act(input_tensor)  # deep_learningクラスのactメソッドを呼び出し
#         steering_angle = output.item()  # 予測値を取得（ラジアン単位）
#         return steering_angle

#     def publish_cmd_vel(self, steering_angle):
#         """推論した舵角を /cmd_vel にパブリッシュ"""
#         cmd_msg = Twist()
#         cmd_msg.linear.x = self.linear_speed  # 前進速度は一定
#         cmd_msg.angular.z = steering_angle    # 予測したハンドリングを適用
#         self.cmd_vel_pub.publish(cmd_msg)