import rclpy
from rclpy.node import Node

from inter_crampc.msg import Vec, VecArray
from nav_msgs.msg import Odometry

class PathGenerator(Node):
    """The Path Generator Class for ROS2. """
    def __init__(self):
        super().__init__('path_generator')
        self.declare_parameter('traj_file', '/home/turtle/ros2_ws/src/cRAMPC/config/segment_0.csv')
        self.declare_parameter('ref_type', 'traj')

        self.get_logger().info('Path Generator node has been started.')
        
        self.debug =True

        if self.get_parameter('ref_type').get_parameter_value().string_value == 'ref':
            self.path_pub = self.create_publisher(Vec, '/trajectory', 10)
        else:
            self.path_pub = self.create_publisher(VecArray, '/trajectory', 10)

        self.odom_sub = self.create_subscription(Odometry, '/ekf_odom', self.odom_callback, 10)

        self.path = []
        self.file_path = self.get_parameter('traj_file').get_parameter_value().string_value

        self.current_position = [0.0, 0.0]
        self.last_index = 0

        self.load_trajectory_from_file()

    def odom_callback(self, msg):
        self.current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.publish_path()

    def publish_path(self):
        path_to_publish = None
        if self.path.__class__.__name__ == 'VecArray':
            # Check if we are close to the last_index point in the path
            path_to_publish = VecArray(array=[])
            distance_to_goal = ((self.current_position[0] - self.path.array[self.last_index].data[0]) ** 2 +
                                (self.current_position[1] - self.path.array[self.last_index].data[1]) ** 2) ** 0.5
            if distance_to_goal < 0.1 and self.last_index + 10 < len(self.path.array) - 1:
                self.last_index += 1
                path_to_publish.array = self.path.array[self.last_index:self.last_index + 10]
            elif distance_to_goal < 0.1 and self.last_index < len(self.path.array) - 1:
                path_to_publish.array = self.path.array[self.last_index:]
            elif distance_to_goal < 0.1 and self.last_index == len(self.path.array) - 1:
                self.get_logger().info('Reached the goal!')
                rclpy.shutdown()
            else:
                path_to_publish.array = self.path.array[self.last_index:self.last_index + 10]
        else:
            distance_to_goal = ((self.current_position[0] - self.path.data[0]) ** 2 +
                                (self.current_position[1] - self.path.data[1]) ** 2) ** 0.5
            if distance_to_goal < 0.1:
                self.get_logger().info('Reached the goal!')
                rclpy.shutdown()
            path_to_publish = self.path
        self.path_pub.publish(path_to_publish)

    def load_trajectory_from_file(self):
        """Load the trajectory from a file."""
        if self.file_path == 'None':
            self.get_logger().info('No trajectory file provided. Sending to default goal.')
            self.path = Vec()
            self.path.data = [5.0, 5.0]  # Default goal
            self.publish_path()

        else:
            self.path = VecArray(array=[])
            with open(self.file_path, 'r') as f:
                # Skip the header
                next(f)
                for line in f:
                    x, y, _, _, _, _ = map(float, line.strip().split(','))
                    self.path.array.append(Vec(data=[x, y]))
            if self.debug:
                self.get_logger().info(f'Trajectory loaded from {self.file_path} with {len(self.path.array)} points.')


def main(args=None):
    rclpy.init(args=args)
    path_generator = PathGenerator()
    rclpy.spin(path_generator)
    path_generator.destroy_node()
    rclpy.shutdown()
