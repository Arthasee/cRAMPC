import rclpy
from rclpy.node import Node

# from plant_robot.msg import OdometryTwoDim
# from plant_robot.msg import RobotTrajectory

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry_msgs.msg import Pose2D


import numpy as np
import matplotlib.pyplot as plt
import collections
from math import cos, sin, sqrt,pi,atan2

fig = plt.figure(figsize=(18, 9))
fig.canvas.manager.set_window_title('Plots')
plt.ion()

class SimPlotNode(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("sim_plt") # MODIFY NAME

        self.declare_parameter("deque_len",20000)
        deque_len = self.get_parameter("deque_len").value

        self.x_ = collections.deque([0.0],maxlen=deque_len)
        self.y_ = collections.deque([0.0],maxlen=deque_len)
        self.theta_ = collections.deque([0.0],maxlen=deque_len)
        self.time_ = collections.deque([0.0],maxlen=deque_len)

        self.last_x_ = 0.0
        self.last_y_ = 0.0
        self.last_theta_ = 0.0

        self.last_ref_x_ = 0.0
        self.last_ref_y_ = 0.0
        self.last_ref_theta_ = 0.0

        self.last_ekf_x_ = 0.0
        self.last_ekf_y_ = 0.0
        self.last_ekf_theta_ = 0.0

        self.last_vx_ = 0.0
        self.last_omega_ = 0.0

        self.ref_x_ = collections.deque([0.0],maxlen=deque_len)
        self.ref_y_ = collections.deque([0.0],maxlen=deque_len)
        self.ref_theta_ = collections.deque([0.0],maxlen=deque_len)

        self.ekf_x_ = collections.deque([0.0],maxlen=deque_len)
        self.ekf_y_ = collections.deque([0.0],maxlen=deque_len)
        self.ekf_theta_ = collections.deque([0.0],maxlen=deque_len)

        self.vx_ = collections.deque([0.0],maxlen=deque_len)
        self.omega_ = collections.deque([0.0],maxlen=deque_len)
    
        # Subscription to Vicon Ground Truth Pose

        self.state_sub_ = self.create_subscription(
            PoseStamped, "donatello/donatello",
            self.update_state_callback,1)
        
        # Subscription to Reference Trajectory Pose
        
        self.ref_sub_ = self.create_subscription(
            Pose2D, "trajectory",
            self.update_ref_callback,1)
        
        # Subscription to EKF Odometry Pose

        self.ekf_sub_ = self.create_subscription(
            Odometry, "odom_ekf",
            self.update_ekf_callback,1)
        
        # Subscription to Command Velocity

        self.cmd_sub_ = self.create_subscription(
            TwistStamped, "cmd_vel",
            self.update_cmd_callback,1)
        
        self.plot_timer_ = self.create_timer(1/10, self.plot_callback)
        self.get_logger().info("Plot is running")
        plt.show()
    
    def update_state_callback(self, in_msg: PoseStamped):
        self.last_x_=in_msg.pose.position.x
        self.last_y_=in_msg.pose.position.y
        self.last_theta_= 2*np.arctan2(in_msg.pose.orientation.z, in_msg.pose.orientation.w)

    def update_ref_callback(self, in_msg: Pose2D):
        self.last_ref_x_ = in_msg.x
        self.last_ref_y_ = in_msg.y
        self.last_ref_theta_ = in_msg.theta

    def update_ekf_callback(self, in_msg: Odometry):
        self.last_ekf_x_ = in_msg.pose.pose.position.x
        self.last_ekf_y_ = in_msg.pose.pose.position.y
        self.last_ekf_theta_ = 2*np.arctan2(in_msg.pose.pose.orientation.z, in_msg.pose.pose.orientation.w)

    def update_cmd_callback(self, in_msg: TwistStamped):
        self.last_vx_ = in_msg.twist.linear.x
        self.last_omega_ = in_msg.twist.angular.z

    def plot_callback(self):

        self.x_.append(self.last_x_)
        self.y_.append(self.last_y_)
        self.theta_.append(self.last_theta_)

        self.ref_x_.append(self.last_ref_x_)
        self.ref_y_.append(self.last_ref_y_)
        self.ref_theta_.append(self.last_ref_theta_)

        self.ekf_x_.append(self.last_ekf_x_)
        self.ekf_y_.append(self.last_ekf_y_)
        self.ekf_theta_.append(self.last_ekf_theta_)

        self.vx_.append(self.last_vx_)
        self.omega_.append(self.last_omega_)

        self.time_.append(self.time_[-1] + 1/10)

        plt.clf()

        # Layout: 6-row x 3-col GridSpec, width_ratios=[1,2,2]
        # col 0 (narrow): ax_vx (top half) and ax_omega (bottom half)
        # col 1 (wide):   ax1 X-Y trajectory (full height)
        # col 2 (wide):   ax2/ax3/ax4 x/y/theta vs time
        gs = fig.add_gridspec(6, 3, width_ratios=[1, 2, 2])
        ax_vx    = fig.add_subplot(gs[0:3, 0])
        ax_omega = fig.add_subplot(gs[3:6, 0])
        ax1      = fig.add_subplot(gs[0:6, 1])
        ax2      = fig.add_subplot(gs[0:2, 2])
        ax3      = fig.add_subplot(gs[2:4, 2])
        ax4      = fig.add_subplot(gs[4:6, 2])

        # ax_vx: linear velocity command vs time
        ax_vx.plot(self.time_, self.vx_, 'b-', linewidth=1.0, label=r'$v_x$')
        ax_vx.set_ylabel(r'$v_x$ (m/s)')
        ax_vx.set_xlim(self.time_[0], self.time_[-1])
        ax_vx.legend(fontsize=7, loc='upper left')
        ax_vx.grid(True)

        # ax_omega: angular velocity command vs time
        ax_omega.plot(self.time_, self.omega_, 'r-', linewidth=1.0, label=r'$\omega$')
        ax_omega.set_xlabel('Time (s)')
        ax_omega.set_ylabel(r'$\omega$ (rad/s)')
        ax_omega.set_xlim(self.time_[0], self.time_[-1])
        ax_omega.legend(fontsize=7, loc='upper left')
        ax_omega.grid(True)

        # ax1: X-Y trajectory
        ax1.plot(self.x_, self.y_, 'b-', linewidth=1.5, label='Ground Truth')
        ax1.plot(self.ref_x_, self.ref_y_, 'r--', linewidth=1.0, label='Reference')
        ax1.plot(self.ekf_x_, self.ekf_y_, 'g-', linewidth=1.0, label='EKF Odom')
        # if self.x_ and self.y_ and (abs(self.last_vx_) > 1e-6):
        ax1.quiver(self.x_[-1], self.y_[-1],
                    self.last_vx_*cos(self.theta_[-1]),
                    self.last_vx_*sin(self.theta_[-1]),
                    scale=0.5, color='purple', label='Velocity')
        ax1.relim()
        ax1.autoscale()
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend(fontsize=7, loc='upper left')
        ax1.grid(True)

        # ax2: X vs Time
        ax2.plot(self.time_, self.x_, 'b-', linewidth=1.0, label=r'$x$')
        ax2.plot(self.time_, self.ref_x_, 'r--', linewidth=1.0, label=r'$x_{ref}$')
        ax2.plot(self.time_, self.ekf_x_, 'g-', linewidth=1.0, label=r'$x_{ekf}$')
        ax2.set_ylabel('x (m)')
        ax2.set_xlim(self.time_[0], self.time_[-1])
        ax2.legend(fontsize=7, loc='upper left')
        ax2.grid(True)

        # ax3: Y vs Time
        ax3.plot(self.time_, self.y_, 'b-', linewidth=1.0, label=r'$y$')
        ax3.plot(self.time_, self.ref_y_, 'r--', linewidth=1.0, label=r'$y_{ref}$')
        ax3.plot(self.time_, self.ekf_y_, 'g-', linewidth=1.0, label=r'$y_{ekf}$')
        ax3.set_ylabel('y (m)')
        ax3.set_xlim(self.time_[0], self.time_[-1])
        ax3.legend(fontsize=7, loc='upper left')
        ax3.grid(True)

        # ax4: Theta vs Time
        ax4.plot(self.time_, self.theta_, 'b-', linewidth=1.0, label=r'$\theta$')
        ax4.plot(self.time_, self.ref_theta_, 'r--', linewidth=1.0, label=r'$\theta_{ref}$')
        ax4.plot(self.time_, self.ekf_theta_, 'g-', linewidth=1.0, label=r'$\theta_{ekf}$')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel(r'$\theta$ (rad)')
        ax4.set_xlim(self.time_[0], self.time_[-1])
        ax4.legend(fontsize=7, loc='upper left')
        ax4.grid(True)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        

def main(args=None):
    rclpy.init(args=args)
    node = SimPlotNode() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
