from DQN import RLAgentWithPER
import gymnasium as gym 

env = gym.make("LunarLander-v3")

agent = RLAgentWithPER(env)
agent.train()