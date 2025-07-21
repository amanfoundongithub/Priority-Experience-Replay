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
        
        