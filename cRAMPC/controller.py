"""The Node to implement cRAMPC library with ROS2."""

import os

from cRAMPC.cRAMPC import CRAMPC
from cRAMPC.cMPC import CMPC
from cRAMPC.cRMPC import CRMPC

from geometry_msgs.msg import TwistStamped

from inter_crampc.msg import PolytopeMsg, Vec, VecArray

import mosek


from nav_msgs.msg import Odometry

import numpy as np

from pycvxset import Polytope
from pycvxset import common as cpy
from pycvxset.common import constants

if constants.DEFAULT_LP_SOLVER_STR == "MOSEK":
    constants.DEFAULT_LP_SOLVER_STR = "CLARABEL"
    constants.DEFAULT_SOCP_SOLVER_STR = "CLARABEL"
    constants.DEFAULT_SDP_SOLVER_STR = "CLARABEL"

import rclpy
from rclpy.node import Node


class Controller(Node):
    """The Controller Class for ROS2."""

    def __init__(self):
        """Initialize the ROS2 Node."""
        super().__init__("controller")
        self.get_logger().info("Controller node has been started.")
        self.get_logger().info("Everyone loves Ice Cream !")

        self.declare_parameter("flavor", "RMPC")
        self.flavor = self.get_parameter("flavor").get_parameter_value().string_value

        self.declare_parameter("Ts", 0.0)
        self.ts = self.get_parameter("Ts").get_parameter_value().double_value

        self.declare_parameter("horizon", 10)
        self.horizon = self.get_parameter("horizon").get_parameter_value().integer_value

        self.declare_parameter("mode", "LQR")
        self.mode = self.get_parameter("mode").get_parameter_value().string_value

        self.declare_parameter("recorder", True)
        self.recorder = self.get_parameter("recorder").get_parameter_value().bool_value

        self.declare_parameter("relax", "")
        self.relax = self.get_parameter("relax").get_parameter_value().string_value

        self.declare_parameter("name", "test_mpc")
        self.name = self.get_parameter("name").get_parameter_value().string_value

        self.declare_parameter("verbose", True)
        self.verbose = self.get_parameter("verbose").get_parameter_value().bool_value

        self.declare_parameter("solver", "osqp")
        self.solver = self.get_parameter("solver").get_parameter_value().string_value

        self.declare_parameter("lbx", [-1e4, -10.0])
        self.lbx = self.get_parameter("lbx").get_parameter_value().double_array_value
        self.declare_parameter("ubx", [1e4, 10.0])
        self.ubx = self.get_parameter("ubx").get_parameter_value().double_array_value
        self.declare_parameter("lbu", [-5.0])
        self.lbu = self.get_parameter("lbu").get_parameter_value().double_array_value
        self.declare_parameter("ubu", [5.0])
        self.ubu = self.get_parameter("ubu").get_parameter_value().double_array_value
        self.declare_parameter("lby", [0.0])
        self.lby = self.get_parameter("lby").get_parameter_value().double_array_value
        self.declare_parameter("uby", [0.0])
        self.uby = self.get_parameter("uby").get_parameter_value().double_array_value

        if self.lby == [0] and self.uby == [0]:
            self.yBound = None
        else:
            self.yBound = (np.array(self.lby), np.array(self.uby))

        self.declare_parameter("svd", False)
        self.svd = self.get_parameter("svd").get_parameter_value().bool_value

        self.declare_parameter("ref", "trajectory")
        self.ref_type = self.get_parameter("ref").get_parameter_value().string_value

        self.declare_parameter("lam", 0.999)
        self.lam = self.get_parameter("lam").get_parameter_value().double_value

        self.declare_parameter("lpv_flag", False)
        self.lpv_flag = self.get_parameter("lpv_flag").get_parameter_value().bool_value

        self.declare_parameter("par_filter", "lms")
        self.par_filter = (
            self.get_parameter("par_filter").get_parameter_value().string_value
        )

        self.declare_parameter("Nc", 0)
        self.Nc = self.get_parameter("Nc").get_parameter_value().integer_value

        self.declare_parameter("sigma", 0.95)
        self.sigma = self.get_parameter("sigma").get_parameter_value().double_value

        self.declare_parameter("customJ", "")
        self.customJ = self.get_parameter("customJ").get_parameter_value().string_value

        self.declare_parameter(
            "A_flat",
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
            "B_flat", [0.2, -0.1, 0.0, 0.1, 1.0, 0.0, 0.4, -0.4]
        )  # [0., 1.])
        self.declare_parameter(
            "C_flat", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # [1., 0.])
        self.declare_parameter("Q_flat", [1.0, 0.0, 0.0, 1.0])
        self.declare_parameter("R_flat", [1.0])

        self.declare_parameter("size_x", 2)
        self.size_x = self.get_parameter("size_x").get_parameter_value().integer_value
        self.declare_parameter("size_u", 1)
        self.size_u = self.get_parameter("size_u").get_parameter_value().integer_value
        self.declare_parameter("size_y", 1)
        self.size_y = self.get_parameter("size_y").get_parameter_value().integer_value

        self.declare_parameter("K_flat", [0.0, 0.0])
        self.K = np.reshape(
            self.get_parameter("K_flat").get_parameter_value().double_array_value,
            (self.size_u, self.size_x),
        )

        self.declare_parameter("W_A_flat", [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        self.declare_parameter("W_b_flat", [0.05, 0.05, 0.05, 0.05])
        W_b = self.get_parameter("W_b_flat").get_parameter_value().double_array_value
        self.W = Polytope(
            A=np.reshape(
                self.get_parameter("W_A_flat").get_parameter_value().double_array_value,
                (len(W_b), self.size_x),
            ),
            b=np.array(W_b).reshape((-1, 1)),
        )

        self.declare_parameter("theta_V_flat", [0.0, 0.0, 0.0, 0.0])
        theta_v_flat = (
            self.get_parameter("theta_V_flat").get_parameter_value().double_array_value
        )

        self.cmd_pub = self.create_publisher(TwistStamped, "cmd_vel", 10)

        self.A = np.reshape(
            self.get_parameter("A_flat").get_parameter_value().double_array_value,
            (self.size_x, self.size_x, -1),
        )
        self.B = np.reshape(
            self.get_parameter("B_flat").get_parameter_value().double_array_value,
            (self.size_x, self.size_u, -1),
        )
        self.C = np.reshape(
            self.get_parameter("C_flat").get_parameter_value().double_array_value,
            (self.size_y, self.size_x, -1),
        )

        self.Q = np.reshape(
            self.get_parameter("Q_flat").get_parameter_value().double_array_value,
            (self.size_x, self.size_x),
        )
        self.R = np.reshape(
            self.get_parameter("R_flat").get_parameter_value().double_array_value,
            (self.size_u, self.size_u),
        )
        if not np.array(theta_v_flat).any():
            self.theta = None
        else:
            theta_v = np.reshape(theta_v_flat, (-1, self.A.shape[2]))
            self.theta = Polytope(V=theta_v)

        self.declare_parameter("theta_c_flat", [0.0, 0.0, 0.0, 0.0])
        theta_c_flat = (
            self.get_parameter("theta_c_flat").get_parameter_value().double_array_value
        )
        if not np.array(theta_c_flat).any():
            self.theta_c = None
        else:
            self.theta_c = np.reshape(theta_c_flat, (-1, self.C.shape[2]))
            self.theta_c = Polytope(V=self.theta_c)

        self.constraints = None
        self.E = None

        if not self.K.any():
            self.K = None

        self.options = {
            "Ts": self.ts,
            "K": self.K,
            "solver": self.solver,
            "verbose": self.verbose,
            "relax": self.relax,
            "customJ": self.customJ,
            "svd": self.svd,
            "Nc": self.Nc,
            "sigma": self.sigma,
            "xBound": (np.array(self.lbx), np.array(self.ubx)),
            "uBound": (np.array(self.lbu), np.array(self.ubu)),
            "yBound": self.yBound,
            "W": self.W,
            "theta": self.theta,
            "theta_c": self.theta_c,
            "lpv_flag": self.lpv_flag,
            "par_filter": self.par_filter,
            "lam": self.lam,
            "E": self.E,
            "name": self.name,
            "ref": self.ref_type,
        }

        self.last_theta = 0.0

        npzfile = np.load(os.path.join('src/cRAMPC/config', 'system_matrices.npz'))
        A, B = npzfile['arr_0'], npzfile['arr_1']
        C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        C = np.stack([C, np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4))], axis=2)
        W = Polytope(
                        A=np.block([[np.eye(4)], [-np.eye(4)]]), b=0.001 * np.ones((8, 1))
                    )

        E = Polytope(
                        A=np.vstack((np.eye(2), -np.eye(2))),
                        b=np.concatenate((np.ones(2) * 0.01, np.ones(2) * 0.01)),
                    )

        Theta_c = Polytope()

        max_p, min_p = npzfile['arr_2'][:,1], npzfile['arr_2'][:,0]

        Theta = Polytope(A = np.block([[np.eye(3)], [-np.eye(3)]]), b=np.block([np.array(max_p), -np.array(min_p)])) # np.block([np.array(max_p), -np.array(min_p)])

        Q, R = np.eye(4), np.eye(2) #np.diag([1/(0.46**2), 1/(1.90**2)])
        npz2file = np.load(os.path.join('src/cRAMPC/config', 'offline_matrices.npz'))
        K, P = npz2file['arr_0'], npz2file['arr_1']

        opt = {
            'K': K,
            "solver": 'osqp',
            "verbose": False,
            "svd": False,
            "xBound": (np.array([-1.90/30, -0.46, -0.01, -1.90]), np.array([1.90/30, 0.46, 0.01, 1.90])),
            "uBound": (np.array([-0.46, -1.90]), np.array([0.46, 1.90])),
            "name": 'tet_mpc',
            "W": W,
            'E': E,
            "theta": Theta,
            "lam": 0.98,
            'par_filter': 'lms',
            'ref': 'trajectory',
        }

        # -------------SHOULD BE DELETED AFTER TESTING PHASE------------- #

        if self.flavor == "MPC":
            self.controller = CMPC(
                {"A": A[:,:,0], "B": B[:,:,0], "C": C[:,:,0]},
                self.Q,
                self.R,
                self.horizon,
                self.options,
            )
        elif self.flavor == "RMPC":
            self.controller = CRMPC(
                {"A": A, "B": B, "C": C},
                Q,
                R,
                self.horizon,
                opt,
            )
        elif self.flavor == 'RAMPC':
            self.controller = CRAMPC({'A': A, 'B': B, 'C': C},
                                     Q, R, self.horizon, opt)

        if self.recorder:
            self.pub_tube = self.create_publisher(PolytopeMsg, "tube_set", 10)
            self.last_idx = (self.controller.N + 1) * self.controller.n

        self.controller.initialize(self.mode, self.constraints)
        self.controller.P = P

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

        ref_type = self.options["ref"] if self.options.get("ref") is not None else ""
        if ref_type.lower() in ["trajectory", "traj"]:
            self.sub_ref = self.create_subscription(
                VecArray, "trajectory", self.ref_callback, 10
            )
        else:
            self.sub_ref = self.create_subscription(
                Vec, "trajectory", self.ref_callback, 10
            )
        self.ref = None

        self.sub_odom = self.create_subscription(
            Odometry, "odom_ekf", self.odom_callback, 10
        )
        self.curr_x = None
        self.get_logger().info("initialization done !")

        self.timer = self.create_timer(1 / 30, self.timer_callback)

    def ref_callback(self, msg):
        """Receive reference trajectory or setpoint."""
        self.get_logger().info(f'{msg.__class__.__name__}')
        if msg.__class__.__name__ == "VecArray":
            self.get_logger().info("Received reference trajectory:")
            self.ref = []
            self.ref = np.array([vec.data for vec in msg.array]).reshape(-1, 1)
        elif msg.__class__.__name__ == "Vec":
            self.ref = msg.data
        # self.ref = np.array(self.ref).reshape((-1, 1))

    def odom_callback(self, msg):
        """Receive current state from odometry."""
        # Calculate theta from quaternion
        theta = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        d_theta = self.last_theta - theta if self.last_theta is not None else 0.0
        self.last_theta = theta
        self.curr_x = np.array([d_theta, msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])

    def timer_callback(self):
        """Solve the MPC problem and send the command."""
        if self.curr_x is not None:
            self.controller.solve(self.curr_x, self.ref)
            msg = TwistStamped()
            if self.controller.u_star.shape[0] > 1 or self.controller.u_star.shape[1] > 1:
                msg.twist.linear.x = float(self.controller.u_star[0])
                msg.twist.angular.z = float(self.controller.u_star[1])
            else:
                msg.twist.linear.x = float(self.controller.u_star)
            msg.header.stamp = self.get_clock().now().to_msg()
            if self.recorder:
                # TODO - clean this and add alpha variation
                self.last_idx = (self.controller.N+1) * self.controller.n
                self.last_idx = self.last_idx + self.controller.m * self.controller.N
                if self.controller.track:
                    self.last_idx = self.last_idx + self.controller.m * self.controller.sym.ua.shape[1]
                self.tube_set = PolytopeMsg()
                tube_a = VecArray(array=[])
                tube_b = Vec()
                tube_a = self.ta
                tube_b.data = self.controller.V.b.tolist()
                self.tube_set.a.array = tube_a.array
                self.tube_set.b.array.append(tube_b)
                alpha = self.controller.sol["x"][
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
