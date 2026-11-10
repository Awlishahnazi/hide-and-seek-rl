import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    Actor network (Policy).
    Takes observation (grid + vec) and outputs action probabilities.
    """
    def __init__(self, obs_channels, grid_size, vec_size, n_actions):
        super(ActorNetwork, self).__init__()
        
        # CNN for the grid
        self.conv1 = nn.Conv2d(obs_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the flattened size after CNN
        # Assuming grid_size is 11x11, after two convs with padding=1, it stays 11x11
        self.flatten_size = 32 * grid_size * grid_size
        
        # MLP for the vector
        self.fc_vec = nn.Linear(vec_size, 32)
        
        # Combined MLP
        self.fc1 = nn.Linear(self.flatten_size + 32, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, obs):
        grid = obs["grid"]
        vec = obs["vec"]
        
        # Process grid
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Process vec
        v = F.relu(self.fc_vec(vec))
        
        # Concatenate
        combined = torch.cat((x, v), dim=1)
        
        # Final MLP
        combined = F.relu(self.fc1(combined))
        action_logits = self.fc2(combined)
        
        # Return logits (the training loop will apply softmax)
        return action_logits

class CriticNetwork(nn.Module):
    """
    Critic network (Value function).
    In MAPPO, the critic gets the global state (or a richer observation).
    For now, we'll use the same local observation as the actor, 
    but in a full implementation, you'd feed it a global state.
    """
    def __init__(self, obs_channels, grid_size, vec_size):
        super(CriticNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(obs_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.flatten_size = 32 * grid_size * grid_size
        self.fc_vec = nn.Linear(vec_size, 32)
        
        self.fc1 = nn.Linear(self.flatten_size + 32, 128)
        self.fc2 = nn.Linear(128, 1)  # Outputs a single value

    def forward(self, obs):
        grid = obs["grid"]
        vec = obs["vec"]
        
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        v = F.relu(self.fc_vec(vec))
        
        combined = torch.cat((x, v), dim=1)
        combined = F.relu(self.fc1(combined))
        
        value = self.fc2(combined)
        return value