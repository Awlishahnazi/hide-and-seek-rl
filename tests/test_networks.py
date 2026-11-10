import torch
import numpy as np
from hideseek.mappo.networks import ActorNetwork, CriticNetwork

def test_actor_network_shapes():
    # Dummy observation: batch_size=2, channels=5, grid=11x11, vec=4
    obs = {
        "grid": torch.randn(2, 5, 11, 11),
        "vec": torch.randn(2, 4)
    }
    
    actor = ActorNetwork(obs_channels=5, grid_size=11, vec_size=4, n_actions=6)
    logits = actor(obs)
    
    assert logits.shape == (2, 6)

def test_critic_network_shapes():
    obs = {
        "grid": torch.randn(2, 5, 11, 11),
        "vec": torch.randn(2, 4)
    }
    
    critic = CriticNetwork(obs_channels=5, grid_size=11, vec_size=4)
    value = critic(obs)
    
    assert value.shape == (2, 1)

def test_rollout_buffer_gae():
    from hideseek.mappo.buffer import RolloutBuffer
    buffer = RolloutBuffer()
    
    # Add dummy data for 3 steps
    for _ in range(3):
        buffer.add(obs=None, action=0, log_prob=0, reward=1.0, value=0.5, done=False)
        
    next_val = 0.5
    advantages, returns = buffer.compute_gae(next_val)
    
    assert len(advantages) == 3
    assert len(returns) == 3
    