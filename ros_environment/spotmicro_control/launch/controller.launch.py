from launch import LaunchDescription
from launch_ros.actions import Node # type: ignore

def generate_launch_description():

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ]
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager"
        ]
    )

    # effort_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=[
    #         "effort_controller",
    #         "--controller-manager",
    #         "/controller_manager"
    #     ]
    # )


    return LaunchDescription([
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
        # effort_controller_spawner
    ])