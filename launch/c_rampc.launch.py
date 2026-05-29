"""Create a launch description for c_rampc. Launch path_generator and c_rampc controller with namespace."""

from launch import LaunchDescription, actions, substitutions

from launch_ros.actions import Node as Node


def generate_launch_description():
    """Generate the launch description for c_rampc."""
    robot_name_arg = actions.DeclareLaunchArgument(
        'robot_name',
        default_value='IDonatello',
        description='Namespace for the robot'
    )
    flavor_arg = actions.DeclareLaunchArgument(
        'flavor',
        default_value='RMPC',
        description='Flavor of the controller - MPC, RMPC or RAMPC'
    )
    horizon_arg = actions.DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='Horizon for the MPC controller'
    )
    mode_arg = actions.DeclareLaunchArgument(
        'mode',
        default_value='LQR',
        description='Mode for the controller - LQR, Volume or Performance'
    )
    recorder_arg = actions.DeclareLaunchArgument(
        'recorder',
        default_value='True',
        description='Whether to record the data or not'
    )
    A_flat_arg = actions.DeclareLaunchArgument(
        'A_flat',
        default_value='[-7.00000000e-01,  0.00000000e+00, -5.00000000e-02,  5.00000000e-02,'
        '-1.00000000e-01,  2.50000000e-01, -1.38777878e-17, -2.50000000e-01, -1.00000000e-01,'
        '-2.50000000e-01,  2.50000000e-01, -1.38777878e-17, -6.00000000e-01,  0.00000000e+00,'
        '-5.00000000e-02,  5.00000000e-02]',
        description='Flattened A matrix for the controller'
    )
    B_flat_arg = actions.DeclareLaunchArgument(
        'B_flat',
        default_value='[ 0.2, -0.1,  0. ,  0.1,  1. ,  0. ,  0.4, -0.4]',
        description='Flattened B matrix for the controller'
    )
    C_flat_arg = actions.DeclareLaunchArgument(
        'C_flat',
        default_value='[1., 0., 0., 0., 0., 0., 0., 0.]',
        description='Flattened C matrix for the controller'
    )
    Q_flat_arg = actions.DeclareLaunchArgument(
        'Q_flat',
        default_value='[1., 0., 1., 0.]',
        description='Flattened Q matrix for the controller'
    )
    R_flat_arg = actions.DeclareLaunchArgument(
        'R_flat',
        default_value='[1.0]',
        description='Flattened R matrix for the controller'
    )
    size_x_arg = actions.DeclareLaunchArgument(
        'size_x',
        default_value='2',
        description='Size of the state vector'
    )
    size_u_arg = actions.DeclareLaunchArgument(
        'size_u',
        default_value='1',
        description='Size of the control vector'
    )
    size_y_arg = actions.DeclareLaunchArgument(
        'size_y',
        default_value='1',
        description='Size of the output vector'
    )

    robot_name = substitutions.LaunchConfiguration('robot_name')
    flavor = substitutions.LaunchConfiguration('flavor')
    horizon = substitutions.LaunchConfiguration('horizon')
    mode = substitutions.LaunchConfiguration('mode')
    recorder = substitutions.LaunchConfiguration('recorder')
    A_flat = substitutions.LaunchConfiguration('A_flat')
    B_flat = substitutions.LaunchConfiguration('B_flat')
    C_flat = substitutions.LaunchConfiguration('C_flat')
    Q_flat = substitutions.LaunchConfiguration('Q_flat')
    R_flat = substitutions.LaunchConfiguration('R_flat')
    size_x = substitutions.LaunchConfiguration('size_x')
    size_u = substitutions.LaunchConfiguration('size_u')
    size_y = substitutions.LaunchConfiguration('size_y')

    return LaunchDescription([
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
            package='c_rampc',
            executable='controller',
            name='controller',
            output='screen',
            parameters=[
                {'flavor': flavor},
                {'horizon': horizon},
                {'mode': mode},
                {'recorder': recorder},
                {'A_flat': A_flat},
                {'B_flat': B_flat},
                {'C_flat': C_flat},
                {'Q_flat': Q_flat},
                {'R_flat': R_flat},
                {'size_x': size_x},
                {'size_u': size_u},
                {'size_y': size_y}
            ]
        ),
        Node(
            namespace=robot_name,
            package='c_rampc',
            executable='path_generator',
            name='path_generator',
            output='screen',
            parameters=[
                {'traj_file': '/home/stream/Personals/Fabio/ros2_ws/src/cRAMPC/config/segment_0.csv'},
                {'ref_type': 'traj'},
                {'ref_point': [5.0, 5.0]}
            ]
        )
    ])
