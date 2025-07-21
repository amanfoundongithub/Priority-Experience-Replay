import numpy as np 

class SumTree:
    """
    A class that implements the sum tree, a variation of segment tree
    to take into account priority of the samples before selecting any 
    of them randomly.
    """
    
    def __init__(self, capacity : int = 1000):
        """
        Henerates a sum tree

        Args:
            capacity (int, optional): The maximum capacity. Defaults to 1000.
        """
        
        self.__capacity = capacity 
        
        # Tree as an array
        self.__tree = np.zeros(2 * capacity - 1) 
        # Explanation:
        # First capacity - 1 => leaves/actual samples
        # Next  capacity     => the tree's structure
        
        self.__data = np.array(capacity, dtype = object)   # Actual data
        self.__write = 0
        self.__no_of_entries = 0
        
    
    def __propogate(self, idx : int, change : int):
        """
        Propogates the change of value on the tree, recursively

        Args:
            idx (int): The index to be updated
            change (int): The change to be added 
        """
        # Get the parent node
        parent = (idx - 1) // 2
        self.__tree[parent] += change 
        
        if parent != 0:
            # Recursively update backwards
            self.__propogate(parent, change) 
            
        
    def __retrieve(self, idx : int, sampling_value : float):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.__tree):
            return idx 

        if sampling_value <= self.__tree[left]:
            return self.__retrieve(left, sampling_value)
        else:
            return self.__retrieve(right, sampling_value - self.__tree[left])
            
    
    def add(self, data : object, priority : float):
        """
        Adds a data to the tree, with the priority

        Args:
            data (object): The object to be added
            priority (float): The priority to be assigned
        """
        idx = self.__write + self.__capacity - 1
        self.__data[self.__write] = data 
        
        self.update(idx, priority)
        
        self.__write += 1
        if self.__write >= self.__capacity:
            self.__write = 0
        
        self.__no_of_entries+= 1
        if self.__no_of_entries >= self.__capacity:
            self.__no_of_entries = self.__capacity
    
    
    def update(self, idx : int, priority : float):
        
        change = priority - self.__tree[idx] 
        self.__tree[idx] = priority
        self.__propogate(idx, change)
        
        
    def find(self, sampling_value : float):
        idx = self.__retrieve(0, sampling_value)
        data_idx = idx - (self.__capacity - 1)
        return (idx, self.__tree[idx], self.__data[data_idx])

    def total(self):
        return self.__tree[0] 
    
        
        