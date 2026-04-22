#Assignee: filippo

from spotmicro.devices.device import Device
from spotmicro.agent.input import Input
import numpy as np
import random 
import math


class SmoothRandomController(Device):

    def __init__(self):

        self._x = 0
        self._previous_x = 0
        self._target_x = 0

        self._y = 0
        self._previous_y = 0
        self._target_y = 0

        self._w = 0
        self._previous_w = 0
        self._target_w = 0

        self._command_step_target = 0
        self._command_step_counter = 0
        self._w_step_target = 0
        self._w_step_counter = 0

    #reads current input
    def read(self, *args, **kwargs) -> Input:
        return Input(self._x, self._y, self._w)

    #uptates accordingly current input
    def update(self, *args, **kwargs):
        self._x = self._smooth(self._x, self._previous_x, self._target_x)
        self._previous_x = self._x

        self._y = self._smooth(self._y, self._previous_y, self._target_y)
        self._previous_y = self._y

        self._w = self._smooth(self._w, self._previous_w, self._target_w)
        self._previous_w = self._w

        if self._change_command():
           self._target_x =  self._sample_command_x()
           self._target_y = self._sample_command_y()

        if self._change_w():
            self._target_w = self._sample_w()

    #smoothing function
    def _smooth(self, prev, trg, alpha=0.15):
        return (1 - alpha) * prev + alpha * trg

    #updates x and y step counter and decides if its time to change the target
    def _change_command(self, mu=200, sigma=50):
        self._command_step_counter += 1

        if self._command_step_counter >= self._command_step_target:
            self._command_step_target = int(np.random.normal(loc=mu, scale=sigma))
            self._command_step_counter = 0
            return True
        
        return False

    #updates w step counter and decides if its time to change the target
    def _change_w(self, mu=100, sigma=50):
        self._w_step_counter += 1

        if self._w_step_counter >= self._w_step_target:
            self._w_step_target = np.random.normal(loc=mu, scale=sigma)
            self._w_step_counter = 0
            return True
        
        return False
    
    #samples a new x
    def _sample_command_x(self):
        return random.uniform(-0.8, 0.8)
    
    #samples a new y
    def _sample_command_y(self):
        return random.choice(-0.4, 0.4)
    
    #samples a new q
    def _sample_w(self):
        return random.uniform((-0.3, 0.3))
