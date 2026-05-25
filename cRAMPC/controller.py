"""The Node to implement cRAMPC library with ROS2."""

import numpy as np

from pycvxset import Polytope

import os
import mosek

from pycvxset import common as cpy
from pycvxset.common import constants

if constants.DEFAULT_LP_SOLVER_STR == 'MOSEK':
    constants.DEFAULT_LP_SOLVER_STR = 'CLARABEL'
    constants.DEFAULT_SOCP_SOLVER_STR = 'CLARABEL'
    constants.DEFAULT_SDP_SOLVER_STR = 'CLARABEL'

import rclpy
from rclpy.node import Node

from inter_crampc.msg import Vec, VecArray
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

# from cRAMPC import CRAMPC
from cRAMPC.cMPC import CMPC
from cRAMPC.cRMPC import CRMPC

# TODO - See how to implement with EPF trajectory tracking controller.


class Controller(Node):
    """The Controller Class for ROS2. """
    def __init__(self):
        super().__init__('controller')
        self.get_logger().info('Controller node has been started.')
        self.get_logger().info('Everyone loves Ice Cream !')

        self.declare_parameter('flavor', 'MPC')
        self.flavor = self.get_parameter('flavor').get_parameter_value().string_value

        self.declare_parameter('horizon', 10)
        self.horizon = self.get_parameter('horizon').get_parameter_value().integer_value

        self.declare_parameter('mode', 'LQR')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        
        self.declare_parameter('A_flat', [1., 1., 0., 1.])
        self.declare_parameter('B_flat', [0., 1.])
        self.declare_parameter('C_flat', [1., 0.])
        self.declare_parameter('Q_flat', [1., 0., 0., 1.])
        self.declare_parameter('R_flat', [.1])

        self.declare_parameter('size_x', 2)
        self.declare_parameter('size_u', 1)
        self.declare_parameter('size_y', 1)

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        self.A = np.reshape(self.get_parameter('A_flat').get_parameter_value().double_array_value,
                            (self.get_parameter('size_x').get_parameter_value().integer_value,
                             self.get_parameter('size_x').get_parameter_value().integer_value))
        self.B = np.reshape(self.get_parameter('B_flat').get_parameter_value().double_array_value,
                            (self.get_parameter('size_x').get_parameter_value().integer_value,
                             self.get_parameter('size_u').get_parameter_value().integer_value))
        self.C = np.reshape(self.get_parameter('C_flat').get_parameter_value().double_array_value,
                            (self.get_parameter('size_y').get_parameter_value().integer_value,
                             self.get_parameter('size_x').get_parameter_value().integer_value))

        self.Q = np.reshape(self.get_parameter('Q_flat').get_parameter_value().double_array_value,
                            (self.get_parameter('size_x').get_parameter_value().integer_value,
                             self.get_parameter('size_x').get_parameter_value().integer_value))
        self.R = np.reshape(self.get_parameter('R_flat').get_parameter_value().double_array_value,
                            (self.get_parameter('size_u').get_parameter_value().integer_value,
                             self.get_parameter('size_u').get_parameter_value().integer_value))
        
        self.constraints = None

        self.options = {
            'solver':'osqp',
            'verbose':True,
            'svd':False,
            'xBound':(np.array([-8, -8]), np.array([8, 8])),
            'uBound':(np.array([-1]),np.array([1])),
            'name':'test_mpc',
            'ref': 'traj'
        }

        A1 = np.array([[-0.7, 0.15], [-0.35, -0.6]])
        A2 = np.array([[-0.75, -0.1], [0.15, -0.65]])
        A3 = np.array([[-0.65, -0.35], [-0.1, -0.55]])

        A0 = 1/3*(A1+A2+A3)

        dA1 = A1 - A0
        dA2 = A2 - A0
        dA3 = A3 - A0

        B1 = np.array([[0.1], [1]])
        B2 = np.array([[0.2], [1.4]])
        B3 = np.array([[0.3], [0.6]])

        B0 = 1/3*(B1+B2+B3)

        dB1 = B1 - B0
        dB2 = B2 - B0
        dB3 = B3 - B0

        theta_v = np.eye(3)#U[:, :2].T @ blocked_delta
        Theta = Polytope(V=theta_v.T)

        A = np.stack([A0, dA1, dA2, dA3], axis=2)
        B = np.stack([B0, dB1, dB2, dB3], axis=2)


        C = np.array([[1, 0]]) # This is enough
        C = np.stack([C, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))], axis=2) # this is correct and can be used to test if c_mask works properly

        Q, R = np.eye(2), np.array([[1]])
        K =np.array([[0.19, 0.34]])

        opt = {
                'K': K,
                'solver':'osqp',
                'verbose':True,
                'svd':False,
                'xBound':(np.array([-1e4, -10]), np.array([1e4, 10])),
                'uBound':(np.array([-5]),np.array([5])),
                'name':'test_mpc'
            }
        curr_x = np.array([2.5, 10.0])

        if self.flavor == 'MPC':
            self.controller = CMPC({'A': self.A, 'B': self.B, 'C': self.C},
                                   self.Q, self.R, self.horizon, self.options)
        elif self.flavor == 'RMPC':
            self.controller = CRMPC({'A': A, 'B': B, 'C': C},
                                    Q, R, self.horizon, opt)
            self.controller.lam= 0.742
            
            self.controller.initialize_uncertainties(Theta)
        # elif self.flavor == 'RAMPC':
        #     self.controller = CRAMPC({'A': self.A, 'B': self.B, 'C': self.C},
        #                              self.Q, self.R, self.horizon, self.options)

        self.controller.initialize("volume", self.constraints)

        ref_type = self.options['ref'] if self.options.get('ref') is not None else ""
        if ref_type.lower() in ['trajectory', 'traj']:
            self.sub_ref = self.create_subscription(VecArray, 'reference',
                                                    self.ref_callback, 10)
        else:
            self.sub_ref = self.create_subscription(Vec, 'reference',
                                                    self.ref_callback, 10)
        self.ref = None

        self.sub_odom = self.create_subscription(Odometry, 'ekf_TOCHECK', self.odom_callback, 10)
        self.curr_x = None

    def ref_callback(self, msg):
        """Receive reference trajectory or setpoint."""
        if msg.__class__.__name__ == 'VecArray':
            self.get_logger().info('Received reference trajectory:')
            self.ref = []
            for i, vec in enumerate(msg.array):
                self.ref.append(vec.data)
        elif msg.__class__.__name__ == 'Vec':
            self.ref = msg.data
            self.get_logger().info(f'Received reference point: {msg.data}')

    def odom_callback(self, msg):
        """Receive current state from odometry."""
        self.curr_x = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.angular.z])


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
