"""The Node to implement cRAMPC library with ROS2."""

import os

from cRAMPC.cMPC import CMPC
from cRAMPC.cRAMPC import CRAMPC
from cRAMPC.cRMPC import CRMPC

from geometry_msgs.msg import TwistStamped, Pose2D, PoseStamped

from inter_crampc.msg import PolytopeMsg, Vec, VecArray

from std_msgs.msg import Bool

from nav_msgs.msg import Odometry

import numpy as np

import casadi as ca

import csv


from pycvxset import Polytope
from pycvxset import common as cpy
from pycvxset.common import constants

if constants.DEFAULT_LP_SOLVER_STR == 'MOSEK':
    constants.DEFAULT_LP_SOLVER_STR = 'CLARABEL'
    constants.DEFAULT_SOCP_SOLVER_STR = 'CLARABEL'
    constants.DEFAULT_SDP_SOLVER_STR = 'CLARABEL'

import rclpy
from rclpy.node import Node


class AngleTracker:
    def __init__(self, axis=(0, 0, 1)):
        """
        Set the main axis of rotation. 
        Default is Z-axis (0, 0, 1). change to (1, 0, 0) for X or (0, 1, 0) for Y.
        """
        self.total_angle = 0.0
        self.last_angle = None
        self.axis = axis

    def update(self, qx, qy, qz, qw):
        sin_half = np.sqrt(qx**2 + qy**2 + qz**2)

        current_angle = 2.0 * np.arctan2(sin_half, qw)

        # This projects the quaternion vector onto your chosen rotation axis
        dot = qx * self.axis[0] + qy * self.axis[1] + qz * self.axis[2]
        if dot < 0:
            current_angle = -current_angle

        if self.last_angle is None:
            self.last_angle = current_angle
            self.total_angle = current_angle
            return self.total_angle

        diff = current_angle - self.last_angle

        if diff > np.pi:
            diff -= 2.0 * np.pi
        elif diff < -np.pi:
            diff += 2.0 * np.pi

        self.total_angle += diff
        self.last_angle = current_angle

        return self.total_angle


class Controller(Node):
    """The Controller Class for ROS2."""

    def __init__(self):
        """Initialize the ROS2 Node."""
        super().__init__('controller')
        self.get_logger().info('Controller node has been started.')
        self.get_logger().info('Everyone loves Ice Cream !')

        self.declare_parameter('flavor', 'RMPC')
        self.flavor = self.get_parameter('flavor').get_parameter_value().string_value

        self.declare_parameter('Ts', 0.0)
        self.ts = self.get_parameter('Ts').get_parameter_value().double_value

        self.declare_parameter('horizon', 10)
        self.horizon = self.get_parameter('horizon').get_parameter_value().integer_value

        self.declare_parameter('mode', 'LQR')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        self.declare_parameter('recorder', True)
        self.recorder = self.get_parameter('recorder').get_parameter_value().bool_value

        self.declare_parameter('relax', '')
        self.relax = self.get_parameter('relax').get_parameter_value().string_value

        self.declare_parameter('name', 'test_mpc')
        self.name = self.get_parameter('name').get_parameter_value().string_value

        self.declare_parameter('verbose', True)
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.declare_parameter('solver', 'osqp')
        self.solver = self.get_parameter('solver').get_parameter_value().string_value

        self.declare_parameter('lbx', [-1e4, -10.0])
        self.lbx = self.get_parameter('lbx').get_parameter_value().double_array_value
        self.declare_parameter('ubx', [1e4, 10.0])
        self.ubx = self.get_parameter('ubx').get_parameter_value().double_array_value
        self.declare_parameter('lbu', [-5.0])
        self.lbu = self.get_parameter('lbu').get_parameter_value().double_array_value
        self.declare_parameter('ubu', [5.0])
        self.ubu = self.get_parameter('ubu').get_parameter_value().double_array_value
        self.declare_parameter('lby', [0.0])
        self.lby = self.get_parameter('lby').get_parameter_value().double_array_value
        self.declare_parameter('uby', [0.0])
        self.uby = self.get_parameter('uby').get_parameter_value().double_array_value

        if self.lby == [0] and self.uby == [0]:
            self.yBound = None
        else:
            self.yBound = (np.array(self.lby), np.array(self.uby))

        self.declare_parameter('svd', False)
        self.svd = self.get_parameter('svd').get_parameter_value().bool_value

        self.declare_parameter('ref_type', 'ref')
        self.ref_type = self.get_parameter('ref_type').get_parameter_value().string_value

        self.declare_parameter('lam', 0.999)
        self.lam = self.get_parameter('lam').get_parameter_value().double_value

        self.declare_parameter('lpv_flag', False)
        self.lpv_flag = self.get_parameter('lpv_flag').get_parameter_value().bool_value

        self.declare_parameter('par_filter', 'lms')
        self.par_filter = (
            self.get_parameter('par_filter').get_parameter_value().string_value
        )

        self.declare_parameter('Nc', 0)
        self.Nc = self.get_parameter('Nc').get_parameter_value().integer_value

        self.declare_parameter('sigma', 0.95)
        self.sigma = self.get_parameter('sigma').get_parameter_value().double_value

        self.declare_parameter('customJ', '')
        self.customJ = self.get_parameter('customJ').get_parameter_value().string_value

        self.declare_parameter(
            'A_flat',
            [
                -7.0e-01,
                0.0e00,
                -5.0e-02,
                5.0e-02,
                -1.0e-01,
                2.5e-01,
                -1.38777878e-17,
                -2.5e-01,
                -1.0e-01,
                -2.5e-01,
                2.5e-01,
                -1.38777878e-17,
                -6.0e-01,
                0.0e00,
                -5.0e-02,
                5.0e-02,
            ],
        )  # [1., 1., 0., 1.])
        self.declare_parameter(
            'B_flat', [0.2, -0.1, 0.0, 0.1, 1.0, 0.0, 0.4, -0.4]
        )  # [0., 1.])
        self.declare_parameter(
            'C_flat', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # [1., 0.])
        self.declare_parameter('Q_flat', [1.0, 0.0, 0.0, 1.0])
        self.declare_parameter('R_flat', [1.0])

        self.declare_parameter('size_x', 2)
        self.size_x = self.get_parameter('size_x').get_parameter_value().integer_value
        self.declare_parameter('size_u', 1)
        self.size_u = self.get_parameter('size_u').get_parameter_value().integer_value
        self.declare_parameter('size_y', 1)
        self.size_y = self.get_parameter('size_y').get_parameter_value().integer_value

        self.declare_parameter('K_flat', [0.0, 0.0])
        self.K = np.reshape(
            self.get_parameter('K_flat').get_parameter_value().double_array_value,
            (self.size_u, self.size_x),
        )

        self.declare_parameter('W_A_flat', [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        self.declare_parameter('W_b_flat', [0.05, 0.05, 0.05, 0.05])
        W_b = self.get_parameter('W_b_flat').get_parameter_value().double_array_value
        self.W = Polytope(
            A=np.reshape(
                self.get_parameter('W_A_flat').get_parameter_value().double_array_value,
                (len(W_b), self.size_x),
            ),
            b=np.array(W_b).reshape((-1, 1)),
        )

        self.declare_parameter('theta_V_flat', [0.0, 0.0, 0.0, 0.0])
        theta_v_flat = (
            self.get_parameter('theta_V_flat').get_parameter_value().double_array_value
        )

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        self.A = np.reshape(
            self.get_parameter('A_flat').get_parameter_value().double_array_value,
            (self.size_x, self.size_x, -1),
        )
        self.B = np.reshape(
            self.get_parameter('B_flat').get_parameter_value().double_array_value,
            (self.size_x, self.size_u, -1),
        )
        self.C = np.reshape(
            self.get_parameter('C_flat').get_parameter_value().double_array_value,
            (self.size_y, self.size_x, -1),
        )

        self.Q = np.reshape(
            self.get_parameter('Q_flat').get_parameter_value().double_array_value,
            (self.size_x, self.size_x),
        )
        self.R = np.reshape(
            self.get_parameter('R_flat').get_parameter_value().double_array_value,
            (self.size_u, self.size_u),
        )
        if not np.array(theta_v_flat).any():
            self.theta = None
        else:
            theta_v = np.reshape(theta_v_flat, (-1, self.A.shape[2]))
            self.theta = Polytope(V=theta_v)

        self.declare_parameter('theta_c_flat', [0.0, 0.0, 0.0, 0.0])
        theta_c_flat = (
            self.get_parameter('theta_c_flat').get_parameter_value().double_array_value
        )
        if not np.array(theta_c_flat).any():
            self.theta_c = None
        else:
            self.theta_c = np.reshape(theta_c_flat, (-1, self.C.shape[2]))
            self.theta_c = Polytope(V=self.theta_c)

        self.constraints = None
        self.E = None
        self.path = None

        if not self.K.any():
            self.K = None

        self.options = {
            'Ts': self.ts,
            'K': self.K,
            'solver': self.solver,
            'verbose': self.verbose,
            'relax': self.relax,
            'customJ': self.customJ,
            'svd': self.svd,
            'Nc': self.Nc,
            'sigma': self.sigma,
            'xBound': (np.array(self.lbx), np.array(self.ubx)),
            'uBound': (np.array(self.lbu), np.array(self.ubu)),
            'yBound': self.yBound,
            'W': self.W,
            'theta': self.theta,
            'theta_c': self.theta_c,
            'lpv_flag': self.lpv_flag,
            'par_filter': self.par_filter,
            'lam': self.lam,
            'E': self.E,
            'name': self.name,
            'ref': self.ref_type,
        }
        self.last_index = 0

        self.last_theta = 0.0

        # npzfile = np.load(os.path.join('src/cRAMPC/config', 'system_matrices.npz'))
        A = np.block([[[np.eye(3)]], [[np.zeros((3, 3))]], [[np.zeros((3, 3))]]])
        A = A.transpose(1, 2, 0).copy()

        B = np.block([[[np.zeros((2, 2))], [0, 1]], [[1, 0], [np.zeros((2, 2))]],
                      [[np.zeros((1, 2))], [1, 0], [np.zeros((1, 2))]]])/30
        B = B.transpose(1, 2, 0).copy()

        C = np.block([[[np.eye(3)]], [[np.zeros((3, 3))]], [[np.zeros((3, 3))]]])
        C = C.transpose(1, 2, 0).copy()

        Q, R = 200*np.diag(np.array([1, 1, 0.8])), 0.5*np.eye(2)
        Theta = Polytope(A=np.block([[np.eye(2)], [-np.eye(2)]]), b=np.ones((4, 1)))

        # Define the MPC parameters
        N = 10  # Prediction horizon

        W = Polytope(
                        A=np.block([[np.eye(3)], [-np.eye(3)]]),
                        b=np.array([0.01, 0.01, 0.01*np.pi/180, 0.01, 0.01, 0.01*np.pi/180])
                    )

        E = Polytope(
                        A=np.block([[np.eye(3)], [-np.eye(3)]]),
                        b=np.array([0.01, 0.01, 0.01*np.pi/180, 0.01, 0.01, 0.01*np.pi/180])
                    )

        # Theta_c = Polytope()

        # max_p, min_p = npzfile['arr_2'][:, 1], npzfile['arr_2'][:, 0]

        npz2file = np.load(os.path.join('src/cRAMPC/config', 'kin_offline_mat.npz'))
        K, P = npz2file['arr_0'], npz2file['arr_1']

        options = {
            'solver': 'osqp',
            'verbose': True,
            'svd': False,
            'xBound': (-np.array([10, 5, 1e4]), np.array([10, 5, 1e4])),
            'uBound': (-np.array([0.46, 1.90]), np.array([0.46, 1.90])),
            'name': 'turtle_controller',
            'W': W,
            'E': E,
            'K': K,
            'lam': 1,
            'theta': Theta,
            'par_filter': 'kf',
            'lpv_flag': True,
            'ref': 'ref',
            'max_delta_th': 0.001
        }

        # -------------SHOULD BE DELETED AFTER TESTING PHASE------------- #

        if self.flavor == 'MPC':
            self.controller = CMPC(
                {'A': A[:, :, 0], 'B': B[:, :, 0], 'C': C[:, :, 0]},
                self.Q,
                self.R,
                self.horizon,
                self.options,
            )
        elif self.flavor == 'RMPC':
            self.controller = CRMPC(
                {'A': A, 'B': B, 'C': C},
                Q,
                R,
                self.horizon,
                options,
            )
        elif self.flavor == 'RAMPC':
            self.controller = CRAMPC({'A': A, 'B': B, 'C': C},
                                     Q, R, N, options)

        if self.recorder:
            self.pub_tube = self.create_publisher(PolytopeMsg, 'tube_set', 10)
            self.last_idx = (self.controller.N + 1) * self.controller.n
        self.file_path = '/home/stream/Personals/Fabio/ros2_ws/src/cRAMPC/config/segment_0.csv'
        self.curr_x = [0., 0., 0.]

        self.controller.initialize(self.mode, self.constraints, P=P)

        if self.recorder:
            self.tube_set = PolytopeMsg()
            tube_a = VecArray(array=[])
            self.ta = [Vec(data=a) for a in self.controller.V.A.tolist()]
            tube_a.array = self.ta
            tube_b = Vec()
            tube_b.data = self.controller.V.b.tolist()
            self.tube_set.a.array = tube_a.array
            self.tube_set.b.array.append(tube_b)
            self.pub_tube.publish(self.tube_set)

        ref_type = self.options['ref'] if self.options.get('ref') is not None else ''
        if ref_type.lower() in ['trajectory', 'traj']:
            self.sub_ref = self.create_subscription(
                VecArray, 'trajectory', self.ref_callback, 10
            )
        else:
            self.sub_ref = self.create_subscription(
                Pose2D, 'trajectory', self.ref_callback, 10
            )
        self.ref = None

        self.start = True
        self.angle_tracker = AngleTracker()
        self.sub_odom = self.create_subscription(
            PoseStamped, 'donatello/donatello', self.odom_callback, 10
        )
        self.pub_vicon = self.create_publisher(Pose2D, 'vicon_pose', 10)
        # self.load_trajectory_from_file()
        self.get_logger().info('initialization done !')

        self.ref_pub = self.create_publisher(Pose2D, 'trajectory', 10)
        self.start_pub = self.create_publisher(Bool, 'cmd_done', 10)
        self.timer = self.create_timer(1. / 30, self.timer_callback)
        self.timer2 = self.create_timer(1.0/30, self.callback_timer2)

    def ref_callback(self, msg: Pose2D):
        """Receive reference trajectory or setpoint."""
        # if msg.__class__.__name__ == 'VecArray':
        #     self.get_logger().info('Received reference trajectory:')
        #     self.ref = []
        #     self.ref = np.array([vec.data for vec in msg.array]).reshape(-1, 1)
        # elif msg.__class__.__name__ == 'Vec':
        #     self.ref = msg.data

        # self.ref = [msg.x, msg.y, msg.theta]
        pass

        # self.ref = list(np.array(self.ref).reshape((-1, 1)).flatten())

    def odom_callback(self, msg):
        """Receive current state from odometry."""
        # Calculate theta from quaternion
        theta = self.angle_tracker.update(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        # theta = 2*np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)
        self.pub_vicon.publish(Pose2D(x=msg.pose.position.x, y=msg.pose.position.y,
                                      theta=msg.pose.orientation.z))  # theta))
        # d_theta = self.last_theta - theta if self.last_theta is not None else 0.0
        self.last_theta = theta
        self.curr_x = np.array([msg.pose.position.x,
                                msg.pose.position.y, msg.pose.orientation.z])  # theta])
        if self.start:
            self.load_trajectory_from_file()
            self.start = False

    def load_trajectory_from_file(self):
        """Load the trajectory from a file."""

        if self.file_path == 'None':
            print('No trajectory file provided. Sending to default goal.')
        else:

            if self.path is None:
                self.path = [[],[],[]]

            with open(self.file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.path[0].append(-float(row['y']) + self.curr_x[0])
                    self.path[1].append(float(row['x']) + self.curr_x[1])
                    self.path[2].append(float(row['theta']) + self.curr_x[2]-np.pi/2)

    def callback_timer2(self):
        self.last_index = (self.last_index + 1)

    def timer_callback(self):
        """Solve the MPC problem and send the command."""
        if self.start:
            start_msg = Bool()
            start_msg.data = True
            self.start_pub.publish(start_msg)
        if not self.start:
            out_msg = Pose2D()
            out_msg.x = self.path[0][self.last_index]
            out_msg.y = self.path[1][self.last_index]
            out_msg.theta = self.path[2][self.last_index]  #np.atan2(out_msg.y - self.curr_x[1], out_msg.x - self.curr_x[0])  # self.path[2][self.last_index]  #
            self.ref_pub.publish(out_msg)
            self.ref = [out_msg.x, out_msg.y, out_msg.theta]
            # if np.linalg.norm(np.array(self.curr_x[:2]) - np.array(self.ref[:2])) <= 0.1:
            #     self.last_index = (self.last_index + 1)
            if self.flavor == 'RAMPC':
                self.controller.solve(ca.DM(self.curr_x), ca.DM(self.curr_x), self.ref)  # [0.5, 0.0, 3.14/4]
            else:
                self.controller.solve(self.curr_x, self.ref)
            msg = TwistStamped()
            MIN_LIN_VEL = 0.07   # TurtleBot minimum linear velocity deadband [m/s]
            MIN_ANG_VEL = 0.07   # TurtleBot minimum angular velocity deadband [rad/s]
            if self.controller.u_star.shape[0] > 1 or self.controller.u_star.shape[1] > 1:
                u = self.controller.u_star.toarray().flatten()
                print(u)
                # if 0.0 < abs(u[0]) < MIN_LIN_VEL:
                #     u[0] = np.sign(u[0]) * MIN_LIN_VEL
                # if 0.0 < abs(u[1]) < MIN_ANG_VEL:
                #     u[1] = np.sign(u[1]) * MIN_ANG_VEL
                msg.twist.linear.x = float(u[0])
                msg.twist.angular.z = float(u[1])
            else:
                u0 = float(self.controller.u_star)
                if 0.0 < abs(u0) < MIN_LIN_VEL:
                    u0 = np.sign(u0) * MIN_LIN_VEL
                msg.twist.linear.x = u0
            msg.header.stamp = self.get_clock().now().to_msg()
            if self.recorder:
                # TODO - clean this and add alpha variation
                self.last_idx = (self.controller.N+1) * self.controller.n
                self.last_idx = self.last_idx + self.controller.m * self.controller.N
                if self.controller.track:
                    self.last_idx = self.last_idx + self.controller.m \
                        * self.controller.sym.ua.shape[1]
                self.tube_set = PolytopeMsg()
                tube_a = VecArray(array=[])
                tube_b = Vec()
                tube_a = self.ta
                tube_b.data = self.controller.V.b.tolist()
                self.tube_set.a.array = tube_a.array
                self.tube_set.b.array.append(tube_b)
                alpha = self.controller.sol['x'][
                    self.last_idx
                    + self.controller.na: self.last_idx
                    + self.controller.na * (self.controller.N + 1)
                ]
                # alpha = np.block([[alpha], [np.zeros((self.controller.na, 1))]])
                alpha = alpha.toarray().tolist()
                for al in alpha:
                    self.tube_set.b.array.append(Vec(data=al))
                self.last_idx = self.last_idx + self.controller.na * (
                    self.controller.N + 1
                )
                self.pub_tube.publish(self.tube_set)
            self.cmd_pub.publish(msg)


def main(args=None):
    """Launch the ROS2 Node."""
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
