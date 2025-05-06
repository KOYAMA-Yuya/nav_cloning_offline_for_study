#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import roslib
roslib.load_manifest('nav_cloning')
import tf.transformations
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist
import csv
import time
import math

class GoalAngleSimulator:
    def __init__(self):
        rospy.init_node('goal_angle_simulator', anonymous=True)

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.amcl_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        rospy.Subscriber('/nav_vel', Twist, self.nav_vel_callback)

        # 角速度の保存用
        self.latest_ang_vel = 0.0

        # CSV読み込み
        csv_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/path/00_02_fix.csv'
        with open(csv_path, 'r') as f:
            self.pos_list = [line.strip().split(',') for line in f]

        self.offset_x = 10.71378
        self.offset_y = 17.17456
        self.goal_offset = 10  # Nステップ後をゴールに

        self.run()

    def nav_vel_callback(self, msg):
        self.latest_ang_vel = msg.angular.z
        print(f"[angular.z] = {self.latest_ang_vel:.4f}")

    def move_robot_pose(self, x, y, theta):
        # Gazebo
        state = ModelState()
        state.model_name = 'turtlebot3'
        state.pose.position.x = x
        state.pose.position.y = y
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quat[0]
        state.pose.orientation.y = quat[1]
        state.pose.orientation.z = quat[2]
        state.pose.orientation.w = quat[3]

        try:
            self.set_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logerr("SetModelState failed: %s" % e)

        # AMCL
        amcl = PoseWithCovarianceStamped()
        amcl.header.stamp = rospy.Time.now()
        amcl.header.frame_id = "map"
        amcl.pose.pose.position.x = x - self.offset_x
        amcl.pose.pose.position.y = y - self.offset_y
        amcl.pose.pose.orientation.x = quat[0]
        amcl.pose.pose.orientation.y = quat[1]
        amcl.pose.pose.orientation.z = quat[2]
        amcl.pose.pose.orientation.w = quat[3]
        amcl.pose.covariance[-1] = 0.01
        self.amcl_pose_pub.publish(amcl)

    def publish_goal(self, x, y):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x - self.offset_x
        goal.pose.position.y = y - self.offset_y
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def run(self):
        rate = rospy.Rate(0.5)  # 2秒ごと（AMCLやmove_baseが安定するように）
        for i in range(len(self.pos_list) - self.goal_offset):
            # 現在位置
            cur = self.pos_list[i]
            x, y, theta = float(cur[1]), float(cur[2]), float(cur[3])
            self.move_robot_pose(x, y, theta)

            rospy.sleep(1.0)  # AMCL反映のため

            # ゴール位置
            goal = self.pos_list[i + self.goal_offset]
            gx, gy = float(goal[1]), float(goal[2])
            self.publish_goal(gx, gy)

            rospy.sleep(2.0)  # move_baseがnav_velを出すのを待つ

            print(f"[Step {i}] 角速度: {self.latest_ang_vel:.4f} rad/s\n")

            rate.sleep()

if __name__ == '__main__':
    GoalAngleSimulator()