import pybullet as p
import numpy as np
from spotmicro.tools.kg_renderer import KG_Renderer
from spotmicro.agent.input import Input

class KinematicGhost():
    
    def __init__(self, renderer: KG_Renderer, dt: float):

        self.renderer = renderer

        self.position = np.zeros(3)
        self.yaw = 0.0

        self.dt = dt

        self.renderer.spawn()

    def reset(self, start_pos, start_quat) -> None:

        self.position = np.array(start_pos, dtype=float)
    
        # estrai yaw dal quaternione iniziale
        x, y, z, w = start_quat
        self.yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
        # aggiorna il renderer con il quaternione del ghost, non del robot
        self.renderer.reset(self.position.copy(), self._yaw_to_quat())
        #self.renderer.reset(start_pos, start_quat)

    def apply_command(self, input: Input) -> None:
        self._integrate(input)
        self.renderer.update(self.position, self._yaw_to_quat(), self.dt)
        

    def _integrate(self, input: Input):
        # legge input
        vx, vy, w = input.vx, input.vy, input.w
        dt = self.dt

        # aggiorna yaw
        self.yaw += w * dt
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        print(f"debug: yaw={self.yaw:.2f}")

        # trasforma velocità nel frame globale
        dx = (- vx * np.cos(self.yaw) + vy * np.sin(self.yaw)) * dt
        dy = (- vx * np.sin(self.yaw) - vy * np.cos(self.yaw)) * dt

        # aggiorna posizione
        self.position[0] += dx
        self.position[1] += dy
        print(f"debug: dx={self.position[0]:.2f} dy={self.position[1]:.2f}")

    def _yaw_to_quat(self):
        # semplice conversione yaw -> quaternion
        cy = np.cos(self.yaw * 0.5)
        sy = np.sin(self.yaw * 0.5)
        q = np.array([0.0, 0.0, sy, cy])
        return q / np.linalg.norm(q)