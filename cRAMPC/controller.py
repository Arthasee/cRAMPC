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

        self.declare_parameter("ref", "ref")
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
 
        A1 = np.array([[-0.7, 0.15], [-0.35, -0.6]])
        A2 = np.array([[-0.75, -0.1], [0.15, -0.65]])
        A3 = np.array([[-0.65, -0.35], [-0.1, -0.55]])

        A0 = np.array([[0.5, 0.2], [-0.1, 0.6]])

        dA1 = np.array([[0.042, 0.], [0.072, 0.03]])
        dA2 = np.array([[0.0015, 0.019], [0.009, 0.035]])
        dA3 = np.array([[0., 0.], [0., 0.]])

        B1 = np.array([[0.1], [1]])
        B2 = np.array([[0.2], [1.4]])
        B3 = np.array([[0.3], [0.6]])

        B0 = np.array([[0.], [0.5]])

        dB1 = np.array([[0.], [0.]])
        dB2 = np.array([[0.], [0.]])
        dB3 = np.array([[0.04], [0.054]])

        theta_v = np.eye(3)
        Theta = Polytope(A=np.block([[np.eye(3)], [-np.eye(3)]]), b=1*np.ones(6,))

        A = np.stack([A0, dA1, dA2, dA3], axis=2)
        B = np.stack([B0, dB1, dB2, dB3], axis=2)

        C = np.array([[1, 0]])
        C = np.stack([C, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))], axis=2)

        Q, R = np.eye(2), np.array([[0.1]])
        K = np.array([[0.017, -0.41]])

        opt = {
            "K": K,
            "solver": self.solver,
            "verbose": False,
            "svd": self.svd,
            "xBound": (np.array([-1e4, -0.3]), np.array([1e4, 1e4])),
            "uBound": (np.array([-1e4]), np.array([1])),
            "name": self.name,
            "W": Polytope(
                A=np.block([[np.eye(2)], [-np.eye(2)]]), b=0.05 * np.ones((4, 1))
            ),
            "theta": Theta,
            "lam": 0.9,
        }

        # -------------SHOULD BE DELETED AFTER TESTING PHASE------------- #

        if self.flavor == "MPC":
            self.controller = CMPC(
                {"A": self.A, "B": self.B, "C": self.C},
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
            self.controller = CRAMPC({'A': self.A, 'B': self.B, 'C': self.C},
                                     self.Q, self.R, self.horizon, self.options)

        if self.recorder:
            self.pub_tube = self.create_publisher(PolytopeMsg, "tube_set", 10)
            self.last_idx = (self.controller.N + 1) * self.controller.n

        self.controller.initialize(self.mode, self.constraints)
        self.controller.P = np.array(([[1.467, 0.207], [0.207, 1.731]]))
        if self.recorder:
            self.tube_set = PolytopeMsg()
            tube_a = VecArray(array=[])
            tube_b = Vec()
            for a in self.controller.V.A.tolist():
                tube_a.array.append(Vec(data=a))
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
            Odometry, "ekf_odom", self.odom_callback, 10
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
            for i, vec in enumerate(msg.array):
                self.ref.append(vec.data)
        elif msg.__class__.__name__ == "Vec":
            self.ref = msg.data

    def odom_callback(self, msg):
        """Receive current state from odometry."""
        self.curr_x = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

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
                for a in self.controller.V.A.tolist():
                    tube_a.array.append(Vec(data=a))
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
