#!/usr/bin/env python3

import rclpy #type: ignore
from rclpy.node import Node #type: ignore
from sensor_msgs.msg import Imu #type: ignore
import numpy as np

class IMUListener(Node):
    def __init__(self):
        super().__init__('imu_listener')
        self.subscription = self.create_subscription(
            Imu,
            '/world/empty/model/differential_drive_robot/link/base_link/sensor/imu_sensor/imu',  # Update this if needed
            self.imu_callback,
            10) 
        
        self.orientation = None

    def imu_callback(self, msg):
        # Extract roll, pitch, yaw from quaternion
        q = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(q.x, q.y, q.z, q.w)
        
        # Log orientation data
        self.get_logger().info(f'Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}')

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, +1.0), -1.0)  # Clamp value
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)