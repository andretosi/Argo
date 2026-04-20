import os
from ament_index_python.packages import get_package_share_directory # type: ignore
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction # type: ignore
from launch.launch_description_sources import PythonLaunchDescriptionSource # type: ignore
from launch_ros.actions import Node # type: ignore
import xacro # type: ignore

def generate_launch_description():
    # 1. Preparo i percorsi dove trovare il file del robot e il file dei controller
    package_name = 'mobile_robot'
    robot_name = 'differential_drive_robot'

    path_model_xacro = os.path.join(
        get_package_share_directory(package_name),
        'models', 'spotmicroai.xacro'
    )

    # Questo è il tuo robot descritto come testo
    robot_description = xacro.process_file(path_model_xacro).toxml()

    # Il tuo file YAML dei controller
    controllers_yaml_path = os.path.join(
        '/home/mario/Desktop/spot_micro_project/spot_micro_with_controller/spotmicro_control/config',
        'controllers.yaml'
    )

    # 2. Lancia GAZEBO con un mondo vuoto
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ]),
        launch_arguments={
            'gz_args': '-r -v -v4 empty.sdf',
            'on_exit_shutdown': 'true'
        }.items()
    )

    # 3. Publisher dello stato del robot (serve per pubblicare le trasformazioni dei link)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        output='screen'
    )

    # 4. Spawna una base sotto il robot (opzionale)
    """
    node_spawn_base_box = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'box_base',
            '-file', os.path.join(
                get_package_share_directory(package_name),
                'models', 'box_base.sdf'
            ),
            '-x', '1.0', '-y', '2.0', '-z', '0.05'
        ],
        output='screen'
    )
    """
    # 5. Spawna il tuo robot nella simulazione
    node_spawn_model_gazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robot_name,
            '-topic', 'robot_description',
            '-x', '1.0', '-y', '2.0', '-z', '0.26'
        ],
        output='screen',
    )

    # 6. Crea un ponte tra ROS 2 e Gazebo
    bridge_params = os.path.join(
        get_package_share_directory(package_name),
        'parameters', 'bridge_parameters.yaml'
    )
    node_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen',
    )

    # 7. Inizializza il sistema di controllo (ros2_control), ma NON carica i controller


    controller_manager = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='ros2_control_node',
                parameters=[
                    {'robot_description': robot_description},
                    controllers_yaml_path
                ],
                output='screen'
            )
        ]
    )

    # 8. Creo il launch description finale
    ld = LaunchDescription()
    ld.add_action(gazebo_launch)
    #ld.add_action(node_spawn_base_box)
    ld.add_action(node_spawn_model_gazebo)
    ld.add_action(node_robot_state_publisher)
    ld.add_action(node_bridge)
    ld.add_action(controller_manager)

    return ld

