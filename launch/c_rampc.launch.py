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
    flavor_arg = actions.DeclareLaunchArgument(
        "flavor",
        default_value="RMPC",
        description="Flavor of the controller - MPC, RMPC or RAMPC",
    )
    horizon_arg = actions.DeclareLaunchArgument(
        "horizon", default_value="5", description="Horizon for the MPC controller"
    )
    mode_arg = actions.DeclareLaunchArgument(
        "mode",
        default_value="LQR",
        description="Mode for the controller - LQR, volume or Performance",
    )
    recorder_arg = actions.DeclareLaunchArgument(
        "recorder",
        default_value="False",
        description="Whether to record the data or not",
    )
    A_flat_arg = actions.DeclareLaunchArgument(
        "A_flat",
        default_value="[-7.00000000e-01,  0.00000000e+00, -5.00000000e-02,  5.00000000e-02,"
        "-1.00000000e-01,  2.50000000e-01, -1.38777878e-17, -2.50000000e-01, -1.00000000e-01,"
        "-2.50000000e-01,  2.50000000e-01, -1.38777878e-17, -6.00000000e-01,  0.00000000e+00,"
        "-5.00000000e-02,  5.00000000e-02]",
        description="Flattened A matrix for the controller",
    )
    B_flat_arg = actions.DeclareLaunchArgument(
        "B_flat",
        default_value="[ 0.2, -0.1,  0. ,  0.1,  1. ,  0. ,  0.4, -0.4]",
        description="Flattened B matrix for the controller",
    )
    C_flat_arg = actions.DeclareLaunchArgument(
        "C_flat",
        default_value="[1., 0., 0., 0., 0., 0., 0., 0.]",
        description="Flattened C matrix for the controller",
    )
    Q_flat_arg = actions.DeclareLaunchArgument(
        "Q_flat",
        default_value="[1., 0., 1., 0.]",
        description="Flattened Q matrix for the controller",
    )
    R_flat_arg = actions.DeclareLaunchArgument(
        "R_flat",
        default_value="[1.0]",
        description="Flattened R matrix for the controller",
    )
    size_x_arg = actions.DeclareLaunchArgument(
        "size_x", default_value="2", description="Size of the state vector"
    )
    size_u_arg = actions.DeclareLaunchArgument(
        "size_u", default_value="1", description="Size of the control vector"
    )
    size_y_arg = actions.DeclareLaunchArgument(
        "size_y", default_value="1", description="Size of the output vector"
    )
    ts_arg = actions.DeclareLaunchArgument(
        "Ts", default_value="0.", description="Sampling time for the controller, if 0, the system is considered continuous"
    )
    relax_arg = actions.DeclareLaunchArgument(
        "relax", default_value="", description="Whether to relax the constraints or not"
    )
    name_arg = actions.DeclareLaunchArgument(
        "name", default_value="", description="Name of the controller, if empty, a random name will be generated"
    )
    verbose_arg = actions.DeclareLaunchArgument(
        "verbose", default_value='True', description="Whether to print verbose output or not"
    )
    solver_arg = actions.DeclareLaunchArgument(
        "solver", default_value="osqp", description="Solver to use for the controller, options are 'qpoases', 'osqp'"
    )
    lbx_arg = actions.DeclareLaunchArgument(
        "lbx", default_value='[0.]', description="Lower bound on the state vector."
    )
    ubx_arg = actions.DeclareLaunchArgument(
        "ubx", default_value='[0.]', description="Upper bound on the state vector."
    )
    lub_arg = actions.DeclareLaunchArgument(
        "lub", default_value='[0.]', description="Lower bound on the control vector."
    )
    uub_arg = actions.DeclareLaunchArgument(
        "uub", default_value='[0.]', description="Upper bound on the control vector."
    )
    lyb_arg = actions.DeclareLaunchArgument(
        "lyb", default_value='[0.]', description="Lower bound on the output vector."
    )
    uyb_arg = actions.DeclareLaunchArgument(
        "uyb", default_value='[0.]', description="Upper bound on the output vector."
    )
    svd_arg = actions.DeclareLaunchArgument(
        "svd", default_value='False', description="Whether to use SVD for the controller or not"
    )
    lam_arg = actions.DeclareLaunchArgument(
        "lam", default_value='0.99', description="Lambda parameter for the controller."
    )
    lpv_flag_arg = actions.DeclareLaunchArgument(
        "lpv_flag", default_value='False', description="Whether the system is LPV or not"
    )
    par_filter_arg = actions.DeclareLaunchArgument(
        "par_filter", default_value='kf', description="Type of parameter filter to use, options are 'lms', 'rls', 'kalman', 'chebyshev'"
    )
    nc_arg = actions.DeclareLaunchArgument(
        "Nc", default_value='0', description="Control horizon for the controller, if 0, it is equal to the horizon"
    )
    sigma_arg = actions.DeclareLaunchArgument(
        "sigma", default_value='0.95', description="Sigma parameter for the controller, used for RMPC and RAMPC"
    )
    customJ_arg = actions.DeclareLaunchArgument(
        "customJ", default_value="", description="Whether to use a custom cost function or not"
    )
    K_flat_arg = actions.DeclareLaunchArgument(
        "K_flat", default_value='[0.0, 0.0]', description="Flattened K matrix for the controller"
    )

    robot_name = substitutions.LaunchConfiguration("robot_name")
    flavor = substitutions.LaunchConfiguration("flavor")
    horizon = substitutions.LaunchConfiguration("horizon")
    mode = substitutions.LaunchConfiguration("mode")
    recorder = substitutions.LaunchConfiguration("recorder")
    A_flat = substitutions.LaunchConfiguration("A_flat")
    B_flat = substitutions.LaunchConfiguration("B_flat")
    C_flat = substitutions.LaunchConfiguration("C_flat")
    Q_flat = substitutions.LaunchConfiguration("Q_flat")
    R_flat = substitutions.LaunchConfiguration("R_flat")
    size_x = substitutions.LaunchConfiguration("size_x")
    size_u = substitutions.LaunchConfiguration("size_u")
    size_y = substitutions.LaunchConfiguration("size_y")

    pkg_share = get_package_share_directory('cRAMPC')
    return LaunchDescription(
        [
            robot_name_arg,
            flavor_arg,
            horizon_arg,
            mode_arg,
            recorder_arg,
            A_flat_arg,
            B_flat_arg,
            C_flat_arg,
            Q_flat_arg,
            R_flat_arg,
            size_x_arg,
            size_u_arg,
            size_y_arg,
            Node(
                namespace=robot_name,
                package='bag_recorder_py',
                executable='odom_sensor',
                name='odom_sensor',
                output='screen',
                parameters=[os.path.join(pkg_share, 'config', 'odom_sensor_params.yaml')],
            ),
            Node(
                namespace=robot_name,
                package='robot_localization',
                executable='ekf_node',
                name='ekf',
                output='screen',
                parameters=[os.path.join(pkg_share, 'config', 'filters_nolidar.yaml')],
                remappings=[
                    ('odometry/filtered', 'odom_ekf'),
                ],
            ),
            Node(
                namespace=robot_name,
                package="cRAMPC",
                executable="controller",
                name="controller",
                output="screen",
                parameters=[
                    {"flavor": flavor},
                    {"horizon": horizon},
                    {"mode": mode},
                    {"recorder": recorder},
                    {"A_flat": A_flat},
                    {"B_flat": B_flat},
                    {"C_flat": C_flat},
                    {"Q_flat": Q_flat},
                    {"R_flat": R_flat},
                    {"size_x": size_x},
                    {"size_u": size_u},
                    {"size_y": 2},
                ],
                on_exit=launch.actions.Shutdown(),
            ),
            # Node(
            #     namespace=robot_name,
            #     package="cRAMPC",
            #     executable="prop",
            #     name='prop',
            #     output='screen',
            #     on_exit=launch.actions.Shutdown(),
            # ),
            Node(
                namespace=robot_name,
                package="cRAMPC",
                executable="path_generator",
                name="path_generator",
                output="screen",
                parameters=[
                    {
                        "traj_file": 'None'#"/home/stream/Personals/Fabio/ros2_ws/src/cRAMPC/config/trajectory_circle.csv"  #"/home/stream/Personals/Fabio/ros2_ws/src/cRAMPC/config/segment_0.csv"
                    },
                    {"ref_type": "ref"},
                    {"ref_point": [0.006, 0.07]},
                    {"odom_type": "velocity_orientation"},
                ],
                on_exit=actions.Shutdown(),

            ),
        ]
    )
