import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

class OffsetResetService(Node):
    def __init__(self):
        super().__init__('offset_reset_service')

        self.declare_parameter("vicon_topic", 'donatello/donatello')
        vicon_topic = self.get_parameter("vicon_topic").value

        self.declare_parameter("republished_topic", 'odom_vicon')

        output_topic = self.get_parameter("republished_topic").value

        self.offset = None
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            vicon_topic,
            self.pose_callback,
            10
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            output_topic,
            10
        )

        self.srv = self.create_service(Trigger, 'reset_offset', self.reset_offset_callback)

        self.last_raw_pose = None


    def reset_offset_callback(self, request, response):
        if self.last_raw_pose is not None:
            self.offset = self.last_raw_pose
            response.success = True
            response.message = f"Offset reset successfully at: {self.offset.pose.position.x}, {self.offset.pose.position.y}, {self.offset.pose.orientation.z}"
        else:
            response.success = False
            response.message = "No pose received yet. Offset reset failed."
        return response

    def pose_callback(self, msg):
        self.last_raw_pose = msg
        if self.offset is None:
            return
        
        msg.pose.position.x -= self.offset.pose.position.x
        msg.pose.position.y -= self.offset.pose.position.y
        msg.pose.orientation.z -= self.offset.pose.orientation.z
        self.pose_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    offset_reset_service = OffsetResetService()
    rclpy.spin(offset_reset_service)
    offset_reset_service.destroy_node()
    rclpy.shutdown()