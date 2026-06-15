"""Create a launch description for c_rampc. Launch path_generator and c_rampc controller with namespace."""
import launch
from launch import LaunchDescription, actions, substitutions
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node as Node
import os


def generate_launch_description():
    """Generate the launch description for c_rampc."""
    robot_name_arg = actions.DeclareLaunchArgument(
        "robot_name", default_value="IDonatello", description="Namespace for the robot"
    )

    robot_name = substitutions.LaunchConfiguration('robot_name')
    return LaunchDescription([
            robot_name_arg,
            Node(
                namespace=robot_name,
                package='cRAMPC',
                executable='plotter',
                name='plot_node',
                output='screen',
            ),
    ])