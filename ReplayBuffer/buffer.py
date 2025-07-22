import numpy as np 

from SumTree import SumTree

class PriorityExperienceReplayBuffer:
    
    def __init__(self, capacity : int = 5000, alpha : float = 0.6, min_priority : float = 1e-5):
        
        self.__tree = SumTree(capacity)
        self.__capacity = capacity
        
        self.__alpha = alpha 
        self.__eps = min_priority
        
    
    def add(self, transistion : object, priority : float):
        
        priority = abs(priority)
        priority = (priority + self.__eps) ** self.__alpha
        
        self.__tree.add(transistion, priority)
    
    def sample(self, batch_size : int = 64, beta : float = 0.6):
        
        total = self.__tree.total() 
        segment = total / batch_size
        
        batch      = []
        indexes    = []
        priorities = []
    
        for i in range(batch_size):
            
            # Sampling value for random priority assignment
            sampling_value = np.random.uniform(i * segment, (i + 1) * segment)
            
            # Get the item from the tree
            idx, priority, data = self.__tree.find(sampling_value)
            
            # Append
            batch.append(data)
            indexes.append(idx)
            priorities.append(priority)
        
        