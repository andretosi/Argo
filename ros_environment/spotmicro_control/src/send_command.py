#!/usr/bin/env python3

import rclpy # type: ignore
from rclpy.node import Node # type: ignore

# Import messaggi
from std_msgs.msg import Float64MultiArray # type: ignore
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # type: ignore
from sensor_msgs.msg import JointState, Imu # <--- NECESSARIO PER LEGGERE LA POSIZIONE #type: ignore
import numpy as np
import math



class AdvancedRobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # ---------------------------------------------------------
        # 1. PUBLISHER: Per inviare comandi (come prima)
        # ---------------------------------------------------------
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # ---------------------------------------------------------
        # 2. SUBSCRIBER: Per leggere lo stato del robot
        # ---------------------------------------------------------
        # Si iscrive al topic standard /joint_states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # ---------------------------------------------------------
        # 3. SUBSCRIBER: Per leggere l'orientazione del robot dall'IMU
        # ---------------------------------------------------------
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',  
            self.imu_callback,
            10) 
        
        self.orientation = None

        # Dizionario per salvare l'ultimo stato letto (per consultazione rapida)
        # Formato: 
        #   { 'joint_name': {'position': float, 'velocity': float, 'effort': float}, ... }
        self.last_known_state = {}

        # array per salvare l'ultima orientazione letta (roll, pitch, yaw)
        self.last_orientation = [0.0, 0.0, 0.0]
        self.last_linear_acceleration = [0.0, 0.0, 0.0]
        self.last_angular_velocity = [0.0, 0.0, 0.0]

        # 

        # Lista dei giunti (stessa di prima)
        self.joint_names = [
            'front_left_shoulder', 'front_left_leg', 'front_left_foot',
            'front_right_shoulder', 'front_right_leg', 'front_right_foot',
            'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot',
            'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot'
        ]

        # Timer per demo (giusto per farti vedere che funziona)
        self.timer = self.create_timer(2.0, self.demo_loop)
        self.demo_step = 0

    # -------------------------------------------------------------
    # FUNZIONE 1: INVIARE TRAIETTORIA (SETTER)
    # -------------------------------------------------------------
    def move_joints(self, target_positions, duration_sec=1.0):
        """
        Invia un comando di movimento a tutti i giunti.
        :param target_positions: Lista di float (deve essere lunga 12)
        :param duration_sec: Tempo in secondi per raggiungere la posizione

        """
        if len(target_positions) != len(self.joint_names):
            self.get_logger().error(f"Errore: Ricevute {len(target_positions)} posizioni, attese {len(self.joint_names)}")
            return

        msg = JointTrajectory()
        '''
        formato del tipo di dato JointTrajectory:
            string[] joint_names, [take this value from self.joint_names]
            JointTrajectoryPoint[] points
        
        formato del tipo di dato JointTrajectoryPoint:
            double[] positions,
            double[] velocities,
            double[] accelerations,
            double[] effort,
            Duration time_from_start

        formato del tipo di dato Duration:
            int32 sec, [integer number of seconds]
            uint32 nanosec [range from 0 a 999 999 999 nanoseconds]

        Duration defines a period between two time points. It is comprised of a
        seconds component and a nanoseconds component.   
        '''
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9) 
        #this line is to convert the fractional part of duration_sec into nanoseconds

        msg.points.append(point)
        
        self.trajectory_pub.publish(msg)
        self.get_logger().info(f"Comando inviato. Target: {target_positions[:3]}... (troncato)")

    # -------------------------------------------------------------
    # FUNZIONE 2: LEGGERE DATI DEI GIUNTI (CALLBACK + GETTER)
    # -------------------------------------------------------------
    def joint_state_callback(self, msg):
        """
        Questa funzione viene chiamata automaticamente ogni volta che
        il robot pubblica il suo stato su /joint_states.
        Formato del tipo di dato msg JointState:
            string[] name,
            double[] position,
            double[] velocity,
            double[] effort.
        """
        # Salviamo i dati in un dizionario per trovarli facilmente col nome
        for i, name in enumerate(msg.name):
            if name in self.joint_names: # Filtriamo solo i giunti che ci interessano
                self.last_known_state[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if len(msg.velocity) > i else 0.0,
                    'effort': msg.effort[i] if len(msg.effort) > i else 0.0
                }

    def get_joint_info(self, joint_name):
        """
        Restituisce le info correnti di uno specifico giunto.
        """
        if joint_name in self.last_known_state:
            return self.last_known_state[joint_name]
        else:
            self.get_logger().warn(f"Nessun dato ancora ricevuto per {joint_name}")
            return None

    # -------------------------------------------------------------
    # FUNZIONE 3: LEGGERE DATI DELL'IMU(CALLBACK)
    # -------------------------------------------------------------

    def imu_callback(self, msg):
        """
        Questa funzione viene chiamata automaticamente ogni volta che
        il robot pubblica la sua orientazione su /imu

        formato del tipo di dato Imu:
        Quaternion orientation,
        float64[9] orientation_covariance,
        Vector3 angular_velocity,
        float64[9] angular_velocity_covariance,
        Vector3 linear_acceleration,
        float64[9] linear_acceleration_covariance
        """
        # Extract roll, pitch, yaw from quaternion
        q = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(q.x, q.y, q.z, q.w)
        self.last_orientation = [roll, pitch, yaw]

        angular_vel = msg.angular_velocity
        self.last_angular_velocity = [angular_vel.x, angular_vel.y, angular_vel.z]

        linear_acc = msg.linear_acceleration
        self.last_linear_acceleration = [linear_acc.x, linear_acc.y, linear_acc.z]
        
        # Log orientation data
        #print(f"DEBUG - IMU -> Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
        #print(f"DEBUG - IMU -> Angular Velocity: {self.last_angular_velocity}")
        #print(f"DEBUG - IMU -> Linear Acceleration: {self.last_linear_velocity}")
        #self.get_logger().info(f'Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}')

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
        # 1. Roll (Rotazione sull'asse X - Inclinazione laterale)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        
        # 2. Pitch (Rotazione sull'asse Y - Inclinazione avanti/indietro)
        t2 = +2.0 * (w * y - z * x)
        # Protezione contro errori di approssimazione in virgola mobile
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        
        # 3. Yaw (Rotazione sull'asse Z - Rotazione a destra/sinistra)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return roll, pitch, yaw

    def get_orientation(self):
        """
        Restituisce l'ultima orientazione letta dall'IMU come (roll, pitch, yaw).
        """
        return self.last_orientation

    def get_angular_velocity(self):
        """
        Restituisce l'ultima velocità angolare letta dall'IMU come (x, y, z).
        """
        return self.last_angular_velocity

    def get_linear_acceleration(self):
        """
        Restituisce l'ultima accelerazione lineare letta dall'IMU come (x, y, z).
        """
        return self.last_linear_acceleration

    # -------------------------------------------------------------
    # DEMO LOOP
    # -------------------------------------------------------------
    def demo_loop(self):
        # ESEMPIO DI UTILIZZO
        
        # 1. Leggiamo lo stato attuale di un giunto (es. spalla anteriore sinistra)
        joint_info = self.get_joint_info('front_left_shoulder')
        orientation_info = self.get_orientation()
        linear_acc_info = self.get_linear_acceleration()
        angular_vel_info = self.get_angular_velocity()

        if joint_info:
            print(f"DEBUG - front_left_shoulder -> Pos: {joint_info['position']:.3f}, Vel: {joint_info['velocity']:.3f}, effort: {joint_info['effort']:.3f}")
        if orientation_info:
            print(f"DEBUG - Orientation -> Roll: {orientation_info[0]:.2f}, Pitch: {orientation_info[1]:.2f}, Yaw: {orientation_info[2]:.2f}")
        if linear_acc_info:
            print(f"DEBUG - Linear Acceleration -> X: {linear_acc_info[0]:.2f}, Y: {linear_acc_info[1]:.2f}, Z: {linear_acc_info[2]:.2f}")
        if angular_vel_info:
            print(f"DEBUG - Angular Velocity -> X: {angular_vel_info[0]:.2f}, Y: {angular_vel_info[1]:.2f}, Z: {angular_vel_info[2]:.2f}")

        # 2. Inviamo un comando diverso a seconda dello step
        if self.demo_step % 2 == 0:
            # Posizione A: tutti a 0.0
            targets = [0.0] * 12
            self.move_joints(targets, duration_sec=0.01)
        else:
            # Posizione B: tutti a 0.3
            targets = [0.3] * 12
            self.move_joints(targets, duration_sec=0.01)
            
        self.demo_step += 1

def main(args=None):
    rclpy.init(args=args)
    
    #questo nodo gestisce il robot:
    # - la lettura dello stato del robot (giunti, IMU, ecc.)
    # - l'invio di comandi di movimento

    AgentNode = AdvancedRobotController()
    
    #questo nodo gestisce la simulazione:
    # - mettere in pausa/avviare la simulazione
    # - riposizionare il robot 
    #EnvNode = SimController() 
    
    rclpy.spin(AgentNode)
    AgentNode.destroy_node()
    #EnvNode.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()