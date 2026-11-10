import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict

from hideseek.env.pz_env import HideSeekEnv
from hideseek.mappo.networks import ActorNetwork, CriticNetwork
from hideseek.mappo.buffer import RolloutBuffer

class MAPPOTrainer:
    def __init__(self, env: HideSeekEnv, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        
        # We will separate teams (Hiders vs Seekers)
        self.teams = {
            "hider": [a for a in env.possible_agents if a.startswith("hider")],
            "seeker": [a for a in env.possible_agents if a.startswith("seeker")]
        }
        
        # Shared networks per team (Parameter sharing)
        obs_space = env.observation_space(env.possible_agents[0])
        channels = obs_space["grid"].shape[0]
        grid_size = obs_space["grid"].shape[1]
        vec_size = obs_space["vec"].shape[0]
        n_actions = env.action_space(env.possible_agents[0]).n
        
        self.actors = {
            "hider": ActorNetwork(channels, grid_size, vec_size, n_actions),
            "seeker": ActorNetwork(channels, grid_size, vec_size, n_actions)
        }
        self.critics = {
            "hider": CriticNetwork(channels, grid_size, vec_size),
            "seeker": CriticNetwork(channels, grid_size, vec_size)
        }
        
        self.actor_optimizers = {
            "hider": optim.Adam(self.actors["hider"].parameters(), lr=lr),
            "seeker": optim.Adam(self.actors["seeker"].parameters(), lr=lr)
        }
        self.critic_optimizers = {
            "hider": optim.Adam(self.critics["hider"].parameters(), lr=lr),
            "seeker": optim.Adam(self.critics["seeker"].parameters(), lr=lr)
        }

    def _batch_obs(self, obs_dict: Dict, team_agents: list) -> Dict:
        """Converts a dict of observations into batched tensors."""
        grids = torch.stack([torch.tensor(obs_dict[a]["grid"], dtype=torch.float32) for a in team_agents])
        vecs = torch.stack([torch.tensor(obs_dict[a]["vec"], dtype=torch.float32) for a in team_agents])
        return {"grid": grids, "vec": vecs}

    def collect_rollout(self, num_steps=128):
        # Initialize buffers for both teams
        buffers = {"hider": RolloutBuffer(), "seeker": RolloutBuffer()}
        
        obs, _ = self.env.reset()
        
        for _ in range(num_steps):
            actions_dict = {}
            # Get actions for all agents
            for team, agents in self.teams.items():
                with torch.no_grad():
                    batched_obs = self._batch_obs(obs, agents)
                    logits = self.actors[team](batched_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values = self.critics[team](batched_obs).squeeze(-1)
                    
                for i, agent in enumerate(agents):
                    actions_dict[agent] = actions[i].item()
                    buffers[team].add(
                        obs=obs[agent],
                        action=actions[i].item(),
                        log_prob=log_probs[i].item(),
                        reward=0, # Will fill in after step
                        value=values[i].item(),
                        done=False
                    )
            
            # Step the environment
            next_obs, rewards, terms, truncs, infos = self.env.step(actions_dict)
            done = any(terms.values()) or any(truncs.values())
            
            # Record rewards
            for team, agents in self.teams.items():
                for agent in agents:
                    # The last added entry for this agent gets the reward
                    buffers[team].rewards[-len(agents) + agents.index(agent)] = rewards[agent]
                    buffers[team].dones[-len(agents) + agents.index(agent)] = done
                    
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
                
        return buffers, obs

    def train(self, total_updates=10):
        print("Starting training...")
        for update in range(total_updates):
            t0 = time.time()
            
            # 1. Collect data
            buffers, last_obs = self.collect_rollout(num_steps=128)
            
            # 2. Update both teams
            for team in ["hider", "seeker"]:
                agents = self.teams[team]
                buffer = buffers[team]
                
                with torch.no_grad():
                    batched_obs = self._batch_obs(last_obs, agents)
                    next_values = self.critics[team](batched_obs).squeeze(-1).tolist()
                
                # Compute GAE
                # Note: For team-based rewards, we use the same next_value for all agents in the team
                # but compute GAE per agent trajectory
                advantages, returns = buffer.compute_gae(next_value=next_values[0], gamma=self.gamma, lam=self.lam)
                
                # Convert to tensors
                advantages = torch.tensor(advantages, dtype=torch.float32)
                returns = torch.tensor(returns, dtype=torch.float32)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize
                
                # PPO Update
                for _ in range(self.epochs):
                    # Reconstruct tensors from buffer
                    grids = torch.stack([torch.tensor(o["grid"], dtype=torch.float32) for o in buffer.observations])
                    vecs = torch.stack([torch.tensor(o["vec"], dtype=torch.float32) for o in buffer.observations])
                    obs_tensor = {"grid": grids, "vec": vecs}
                    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
                    
                    # Actor Loss
                    logits = self.actors[team](obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(torch.tensor(buffer.actions))
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                    
                    # Critic Loss
                    values = self.critics[team](obs_tensor).squeeze(-1)
                    critic_loss = nn.MSELoss()(values, returns)
                    
                    # Optimize
                    self.actor_optimizers[team].zero_grad()
                    actor_loss.backward()
                    self.actor_optimizers[team].step()
                    
                    self.critic_optimizers[team].zero_grad()
                    critic_loss.backward()
                    self.critic_optimizers[team].step()
                    
            print(f"Update {update+1}/{total_updates} | Time: {time.time()-t0:.2f}s | Hider Actor Loss: {actor_loss.item():.3f}")