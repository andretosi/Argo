from abc import ABC, abstractmethod

class TrackerRenderer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def spawn(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset(self):
        #TODO maybe not needed
        pass

