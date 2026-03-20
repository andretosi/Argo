#Assignee mirko

from spotmicro.devices.device import Device
from spotmicro.agent.input import Input


class Joystick(Device):
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self):
        raise NotImplementedError("Joystick device is not implemented yet")
=======
    def __init__(self, joystick_ID: int = 0):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() ==0:
            raise RuntimeError ("No joystick found! Please connect a controller")

        self.joy = pygame.joystick.Joystick(joystick_ID)
        self.joy.init()

        #Initialize the Input array
        self._input_vector = np.array([0.00, 0.00, 0.00], dtype = np.float32)

    def apply_deadzone (self, val, threshold = 0.1):
        if abs(val) < threshold:
            return 0.00
        return(val - np.sign(val) * threshold) / (1.0 - threshold)

    def update(self) -> None:
        pygame.event.pump()

        # read the axis position 

        # Test rapido assi (cortesia di gemini)
        #for i in range(self.joy.get_numaxes()):
        #    print(f"Asse {i}: {self.joy.get_axis(i):.2f}", end=" | ")
        #print()

        # in Pygame vertical axis is inverted(-1 is up and +1 is down), so there has been added a minus (-)
    
        vel_x = -self.joy.get_axis(1) # vertical left stick for up and down
        vel_y = -self.joy.get_axis(0) # horizontal left stick for right and left
        vel_w = -self.joy.get_axis(3) # horizontal right stick for rotation {might also be axis 4}

        # as a safe precaution, ther has been added a dead zone
        # it is needed to prevent that the robot moves as result
        # of external disturbances on the sticks
        self._input_vector[0] = self.apply_deadzone (vel_x)
        self._input_vector[1] = self.apply_deadzone (vel_y)
        self._input_vector[2] = self.apply_deadzone (vel_w)
        

    def read(self) -> Input:
        return self._input_vector

    def reset(self) -> None:
        self._input_vector = np.array([0.00, 0.00, 0.00], dtype=np.float32)
>>>>>>> f68f097 (new uppdated joystick based on pygame)
=======
    def __init__(self):
        raise NotImplementedError("Joystick device is not implemented yet")
>>>>>>> dd1cb2c (Revert "Merge branch 'mirko-controller' of github.com:andretosi/aea_spot_micro into tracker_fixes")
