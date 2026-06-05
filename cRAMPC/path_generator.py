import numpy as np

import rclpy
from rclpy.node import Node

from inter_crampc.msg import Vec, VecArray
from nav_msgs.msg import Odometry


class PathGenerator(Node):
    """The Path Generator Class for ROS2."""

    def __init__(self):
        super().__init__("path_generator")
        self.declare_parameter(
            "traj_file",
            "None"  # "/home/stream/Personals/Fabio/ros2_ws/src/cRAMPC/config/segment_0.csv",
        )
        self.declare_parameter("ref_type", "ref")
        self.declare_parameter("ref_point", [0.0])

        self.declare_parameter("initial_position", [2.0, 3.0])

        self.declare_parameter("odom_type", "full")
        self.odom_type = self.get_parameter("odom_type").get_parameter_value().string_value

        self.get_logger().info("Path Generator node has been started.")

        self.debug = True
        self.ref_point = None
        self.done = False
        self.ref_type = self.get_parameter("ref_type").get_parameter_value().string_value
        if self.ref_type == "ref":
            self.path_pub = self.create_publisher(Vec, "trajectory", 10)
            self.ref_point = (
                self.get_parameter("ref_point").get_parameter_value().double_array_value
            )
        elif self.ref_type in ["traj", "trajectory"]:
            self.path_pub = self.create_publisher(VecArray, "trajectory", 10)

        self.odom_sub = self.create_subscription(
            Odometry, "ekf_odom", self.odom_callback, 10
        )

        self.path = None
        self.file_path = (
            self.get_parameter("traj_file").get_parameter_value().string_value
        )

        self.current_position = self.get_parameter("initial_position").get_parameter_value().double_array_value
        self.last_index = 0

        self.load_trajectory_from_file()

    def odom_callback(self, msg):
        if self.odom_type == "full":
            self.current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z,
                                     msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        elif self.odom_type == "position":
            self.current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        elif self.odom_type == "position_orientation":
            self.current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z]
        elif self.odom_type == "velocity":
            self.current_position = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        elif self.odom_type == "velocity_orientation":
            self.current_position = [msg.pose.pose.orientation.z, msg.twist.twist.linear.x]  # , msg.twist.twist.linear.y, msg.twist.twist.angular.z
        elif self.odom_type == "x":
            self.current_position = [msg.pose.pose.position.x]
        if self.done:
            rclpy.shutdown()
        self.publish_path()

    def publish_path(self):
        path_to_publish = None
        if self.done:
            rclpy.shutdown()
        if self.path is None:
            pass
        elif self.path.__class__.__name__ == "VecArray":
            # Check if we are close to the last_index point in the path
            path_to_publish = VecArray(array=[])
            pos = np.array(self.path.array[self.last_index].data)
            current_pos_array = np.array(self.current_position)
            distance_to_goal = np.linalg.norm(pos - current_pos_array)
            if (
                distance_to_goal < 0.1
                and self.last_index + 10 < len(self.path.array) - 1
            ):
                self.last_index += 1
                path_to_publish.array = self.path.array[
                    self.last_index: self.last_index + 10
                ]
            elif distance_to_goal < 0.1 and self.last_index < len(self.path.array) - 1:
                path_to_publish.array = self.path.array[self.last_index:]
            elif distance_to_goal < 0.1 and self.last_index == len(self.path.array) - 1:
                self.get_logger().info("Reached the goal!")
                rclpy.shutdown()
            else:
                path_to_publish.array = self.path.array[
                    self.last_index: self.last_index + 10
                ]
            self.path_pub.publish(path_to_publish)
        else:
            pos = np.array(self.path.data)
            current_pos_array = np.array(self.current_position)
            distance_to_goal = np.linalg.norm(pos - current_pos_array)
            self.get_logger().info(f'distance : {distance_to_goal}')
            if distance_to_goal <= 0.1:
                self.get_logger().info('hello you hsould stop')
                self.done = True
                self.get_logger().info("Reached the goal!")
                rclpy.shutdown()
            path_to_publish = self.path
            self.path_pub.publish(path_to_publish)

    def load_trajectory_from_file(self):
        """Load the trajectory from a file."""
        if self.file_path == "None" and self.ref_type == 'ref':
            self.get_logger().info(
                "No trajectory file provided. Sending to default goal."
            )
            self.path = Vec()
            self.path.data = self.ref_point
            self.publish_path()

        elif self.file_path != "None" and self.ref_type in ["traj", "trajectory"]:
            self.path = VecArray(array=[])
            with open(self.file_path, "r") as f:
                # Skip the header
                next(f)
                for line in f:
                    if self.odom_type == "full":
                        x, y, theta, vx, vy, omega = map(float, line.strip().split(","))
                        self.path.array.append(Vec(data=[x, y, theta, vx, vy, omega]))
                    elif self.odom_type == "position":
                        x, y, _, _, _, _ = map(float, line.strip().split(","))
                        self.path.array.append(Vec(data=[x, y]))
                    elif self.odom_type == "position_orientation":
                        x, y, theta, _, _, _ = map(float, line.strip().split(","))
                        self.path.array.append(Vec(data=[x, y, theta]))
                    elif self.odom_type == "velocity":
                        _, _, _, vx, vy, omega = map(float, line.strip().split(","))
                        self.path.array.append(Vec(data=[vx, vy, omega]))
                    elif self.odom_type == "velocity_orientation":
                        _, _, theta, vx, vy, omega = map(float, line.strip().split(","))
                        self.path.array.append(Vec(data=[theta, vx]))  # , vy, omega
                    elif self.odom_type == "x":
                        x, _, _, _, _, _ = map(float, line.strip().split(","))
                        self.path.array.append(Vec(data=[x]))
            if self.debug:
                self.get_logger().info(
                    f"Trajectory loaded from {self.file_path} with {len(self.path.array)} points."
                )
        else:
            self.path = None


def main(args=None):
    rclpy.init(args=args)
    path_generator = PathGenerator()
    rclpy.spin(path_generator)
    path_generator.destroy_node()
    rclpy.shutdown()
