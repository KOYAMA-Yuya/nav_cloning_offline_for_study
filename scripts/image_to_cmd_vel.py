#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    try:
        # ROSのImageメッセージをOpenCV画像に変換
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 画像処理（例えば、画像の表示）
        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)  # 1msだけ待機して画面を更新
        
        # 画像に基づいて動作を決める例
        move_cmd = Twist()
        move_cmd.linear.x = 0.5
        move_cmd.angular.z = 0.7

        # /cmd_vel トピックに前進指令を送信
        cmd_vel_pub.publish(move_cmd)
    
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))

def simple_cmd_vel():
    global cmd_vel_pub, bridge
    rospy.init_node('simple_cmd_vel_with_image', anonymous=True)

    # /cmd_vel トピックに指令を出すためのパブリッシャー
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    # /camera/image_raw トピックをサブスクライブ
    rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    
    bridge = CvBridge()  # 画像をOpenCV形式に変換するためのブリッジ
    
    rospy.spin()  # ノードが終了するまで待機

if __name__ == '__main__':
    try:
        simple_cmd_vel()  # シンプルな動作を開始
    except rospy.ROSInterruptException:
        pass