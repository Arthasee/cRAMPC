import numpy as np
import os

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


class PropagationNode(Node):
    def __init__(self):
        super().__init__('prop')

        self.pos = np.array([[0.0], [0.0], [0.0], [0.0]])

        # A0 = np.array([[0.5, 0.2], [-0.1, 0.6]])
        # dA1 = np.array([[0.042, 0.], [0.072, 0.03]])
        # dA2 = np.array([[0.0015, 0.019], [0.009, 0.035]])
        # dA3 = np.array([[0., 0.], [0., 0.]])

        # B0 = np.array([[0.], [0.5]])
        # dB1 = np.array([[0.], [0.]])
        # dB2 = np.array([[0.], [0.]])
        # dB3 = np.array([[0.04], [0.054]])

        # self.A = np.stack([A0, dA1, dA2, dA3], axis=2)
        # self.B = np.stack([B0, dB1, dB2, dB3], axis=2)
        npzfile = np.load(os.path.join('src/cRAMPC/config', 'system_matrices.npz'))
        self.A, self.B = npzfile['arr_0'], npzfile['arr_1']

        self.pub_odom = self.create_publisher(Odometry, 'ekf_odom', 10)
        self.sub_cmd = self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_callback, 10)
        odom_msg = Odometry()
        odom_msg.pose.pose.orientation.z = self.pos[0,0]
        odom_msg.twist.twist.linear.x = self.pos[1,0]
        odom_msg.twist.twist.linear.y = self.pos[2,0]
        odom_msg.twist.twist.angular.z = self.pos[3,0]
        self.pub_odom.publish(odom_msg)

    def cmd_callback(self, msg):
        self.pos = np.einsum('ij,jkl->kl', np.array([[1, 0.8, 0.2, -0.5]]),
                             self.A.transpose(2, 0, 1) @ self.pos + self.B.transpose(2, 0, 1) @ [[msg.twist.linear.x], [msg.twist.angular.z]])
        # self.pos = np.eye(2) @ self.pos + np.array([[1.], [1.]]) * msg.twist.linear.x
        odom_msg = Odometry()
        odom_msg.pose.pose.orientation.z = self.pos[0,0]
        odom_msg.twist.twist.linear.x = self.pos[1,0]
        odom_msg.twist.twist.linear.y = self.pos[2,0]
        odom_msg.twist.twist.angular.z = self.pos[3,0]
        self.pub_odom.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    prop = PropagationNode()
    rclpy.spin(prop)
    prop.destroy_node()
    rclpy.shutdown()
