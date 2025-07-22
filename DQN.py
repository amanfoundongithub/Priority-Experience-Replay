import torch.optim as optim 
import numpy as np 
import torch 

from QNetwork import QNetwork
from ReplayBuffer import PriorityExperienceReplayBuffer


class RLAgentWithPER:
    
    def __init__(self, 
                 state_dim : int,
                 action_dim : int,
                 hidden_dim : int = 128,
                 
                 discount_factor : float = 0.99,
                 batch_size : int = 64,
                 learning_rate : float = 0.001,
                 update_controller : float = 0.005,
                 
                 buffer_size : int = 10000,
                 buffer_alpha : float = 0.6,
                 buffer_beta : float = 0.4,
                 min_priority : float = 1e-5,
                 max_priority : float = 1.0,
                 
                 max_epsilon : float = 1.0,
                 min_epsilon : float = 0.01,
                 eps_decay   : float = 0.99,
                 
                 device : str = "cpu"):
        
        # Main Q Network 
        self.__main_network = QNetwork(state_dim, action_dim, hidden_dim = hidden_dim).to(device)
        self.__action_dim   = action_dim
        
        # Target network (for soft updates) & set it to eval mode 
        self.__target_network = QNetwork(state_dim, action_dim, hidden_dim = hidden_dim).to(device)
        self.__target_network.load_state_dict(self.__main_network.state_dict())
        self.__target_network.eval()
        
        # Set the parameters of the model
        self.__device = device 
        self.__discount_factor = discount_factor
        self.__batch_size = batch_size
        self.__tau = update_controller
        self.__beta = buffer_beta
        
        # Epsilon control parameters (for training)
        self.__epsilon = max_epsilon
        self.__max_eps = max_epsilon
        self.__min_eps = min_epsilon
        self.__eps_decay = eps_decay
        
        # Set the subsidiary buffers & trainers
        self.__buffer = PriorityExperienceReplayBuffer(capacity = buffer_size,
                                                       alpha = buffer_alpha,
                                                       min_priority = min_priority,
                                                       max_priority = max_priority)
        
        self.__trainer = optim.Adam(self.__main_network.parameters(), lr = learning_rate) 
        
        
    def __decide_from_network(self, 
                              state : np.ndarray):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.__device)
        with torch.no_grad():
            return self.__main_network(state).argmax().item()
        
    def act(self, 
            state: np.ndarray, 
            is_train : bool = False):
        
        if is_train:
            if np.random.rand() < self.__epsilon:
                return np.random.randint(low = 0,
                                         high = self.__action_dim - 1)
            
            else: 
                return self.__decide_from_network(state) 
        else: 
            return self.__decide_from_network(state) 
    
    def store(self, 
              transistion : object):
        self.__buffer.add(transistion)
        
    def update(self):
        
        # If buffer is sub-filled
        if len(self.__buffer) < self.__batch_size:
            return 

        # sampler
        batch, indexes, priorities = self.__buffer.sample(batch_size = self.__batch_size,
                                                          beta = self.__beta)
        priorities = torch.tensor(priorities, dtype = torch.float32).to(self.__device)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.__device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.__device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.__device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.__device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.__device)
        
        # Estimate TD error
        Q_values = self.__main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            future_Q_values = self.__target_network(next_states).max(1)[0]
            targets = rewards + self.__discount_factor * future_Q_values * (1 - dones)
        
        td_errors = targets - Q_values
        loss = (td_errors.pow(2) * priorities).mean()
        
        # Trainer
        self.__trainer.zero_grad()
        loss.backward()
        self.__trainer.step()
        
        # Update priorities
        self.__buffer.update(indexes, td_errors.detach().cpu().numpy())
        
        # Soft update
        for target_param, param in zip(self.__target_network.parameters(), self.__main_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Decay epsilon at the end
        self.__epsilon = max(self.__min_eps, self.__epsilon * self.__eps_decay)
        
    
        
    
    