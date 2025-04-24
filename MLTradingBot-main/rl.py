import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Define the Q-network (Deep Q-Network)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.995, batch_size=32, buffer_size=10000):
    
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        

        # Initialize Q-network and target Q-network
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32,device = device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values, dim=1).item()

    def train(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        
        # Compute Q-values from the current Q-network
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q-values from the target network (for the next state)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards +  self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        
        # Backpropagate the loss and update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_experience(self, experience):
        self.buffer.add(experience)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.update_target_network()



    
    
    
"""# Main training loop
def train_dqn_agent(env, agent, episodes=1000):
    
    for episode in range(episodes):
        done = False
        total_reward = 0

        while not done:
            # Select an action
            action = agent.act(state)

            # Take the action, get the new state and reward
            next_state, reward, done, _ = env.step(action)

            # Store experience in replay buffer
            agent.store_experience((state, action, reward, next_state, done))

            # Train the agent
            agent.train()

            state = next_state
            total_reward += reward

        # Update the target network every few episodes
        if episode % 10 == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    agent.save('dqn_agent.pth')
"""