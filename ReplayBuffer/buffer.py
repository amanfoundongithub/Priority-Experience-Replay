import numpy as np 

from SumTree import SumTree

class PriorityExperienceReplayBuffer:
    
    def __init__(self, capacity : int = 5000, alpha : float = 0.6, min_priority : float = 1e-5):
        
        self.__tree = SumTree(capacity)
        self.__capacity = capacity
        
        self.__alpha = alpha 
        self.__eps = min_priority
    
        
        