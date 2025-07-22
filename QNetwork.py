import torch 
import torch.nn as nn 



class QNetwork(nn.Module):
    
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, input : torch.Tensor):
        return self.network(input) 
        