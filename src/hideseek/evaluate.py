import imageio
import torch
import numpy as np
import os
import argparse
from hideseek.env.pz_env import HideSeekEnv
from hideseek.env.render import PygameRenderer
from hideseek.mappo.networks import ActorNetwork

def generate_trained_episode_gif(filename="assets/trained_episode.gif", hider_model_path="results/checkpoints/hider_actor.pth", seeker_model_path="results/checkpoints/seeker_actor.pth"):
    # 1. Initialize Environment and Renderer
    env = HideSeekEnv(map_size=15)
    renderer = PygameRenderer(cell_size=40)
    
    # 2. Load Trained Models
    # Check if models exist
    if not os.path.exists(hider_model_path) or not os.path.exists(seeker_model_path):
        print("Error: Trained models not found. Please run training first.")
        return

    obs_space = env.observation_space(env.possible_agents[0])
    channels = obs_space["grid"].shape[0]
    grid_size = obs_space["grid"].shape[1]
    vec_size = obs_space["vec"].shape[0]
    n_actions = env.action_space(env.possible_agents[0]).n
    
    hider_actor = ActorNetwork(channels, grid_size, vec_size, n_actions)
    seeker_actor = ActorNetwork(channels, grid_size, vec_size, n_actions)
    
    hider_actor.load_state_dict(torch.load(hider_model_path))
    seeker_actor.load_state_dict(torch.load(seeker_model_path))
    
    hider_actor.eval()
    seeker_actor.eval()

    # 3. Run Episode
    obs, _ = env.reset(seed=100) # Fixed seed for reproducibility
    frames = []
    
    for step in range(200): # Max steps
        actions_dict = {}
        
        # Get actions from trained networks
        with torch.no_grad():
            for team, agents in [("hider", [a for a in env.agents if a.startswith("hider")]), 
                                 ("seeker", [a for a in env.agents if a.startswith("seeker")])]:
                
                if not agents: continue
                
                grids = torch.stack([torch.tensor(obs[a]["grid"], dtype=torch.float32) for a in agents])
                vecs = torch.stack([torch.tensor(obs[a]["vec"], dtype=torch.float32) for a in agents])
                batched_obs = {"grid": grids, "vec": vecs}
                
                if team == "hider":
                    logits = hider_actor(batched_obs)
                else:
                    logits = seeker_actor(batched_obs)
                    
                actions = torch.argmax(logits, dim=-1) # Deterministic action (best move)
                
                for i, agent in enumerate(agents):
                    actions_dict[agent] = actions[i].item()
                    
        # Step env
        obs, rewards, terms, truncs, infos = env.step(actions_dict)
        
        # Render (we need to access the core to pass to renderer)
        frame = renderer.render_frame(env.core)
        frames.append(frame)
        
        done = any(terms.values()) or any(truncs.values())
        if done:
            break
            
    renderer.close()
    
    # Save GIF
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Saving GIF to {filename}...")
    imageio.mimsave(filename, frames, fps=10)
    print("Done! Check assets/trained_episode.gif")

if __name__ == "__main__":
    # Ensure you have run `python src/hideseek/train.py --updates 50` first to generate the .pth files!
    generate_trained_episode_gif()