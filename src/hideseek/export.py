import json
import torch
import os
import argparse
from hideseek.env.pz_env import HideSeekEnv
from hideseek.mappo.networks import ActorNetwork
import numpy as np

def export_episode(hider_model_path, seeker_model_path, output_file="demo/replay.json", n_hiders=2, n_seekers=2, vision_radius=5):
    env = HideSeekEnv(n_hiders=n_hiders, n_seekers=n_seekers, vision_radius=vision_radius)
    
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

    obs, _ = env.reset(seed=42)
    
    replay_data = {
        "map_size": env.map_size,
        "initial_grid": env.core.grid.tolist(),
        "steps": []
    }
    
    for step in range(200):
        actions_dict = {}
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
                    
                actions = torch.argmax(logits, dim=-1)
                
                for i, agent in enumerate(agents):
                    actions_dict[agent] = actions[i].item()
                    
        obs, rewards, terms, truncs, infos = env.step(actions_dict)
        
        step_data = {
            "step": step,
            "is_prep_phase": env.core.is_prep_phase,
            "hiders": env.core.hider_positions,
            "seekers": env.core.seeker_positions,
            "grid": env.core.grid.tolist(),
            "rewards": rewards
        }
        replay_data["steps"].append(step_data)
        
        done = any(terms.values()) or any(truncs.values())
        if done:
            break
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(replay_data, f, default=lambda o: int(o) if isinstance(o, (np.integer, np.floating)) else o)
        
    print(f"Replay saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hider_model", type=str, default="results/checkpoints/hider_actor_h2_s2_v5_b2_seed42.pth")
    parser.add_argument("--seeker_model", type=str, default="results/checkpoints/seeker_actor_h2_s2_v5_b2_seed42.pth")
    parser.add_argument("--output", type=str, default="demo/replay.json")
    args = parser.parse_args()
    
    export_episode(args.hider_model, args.seeker_model, args.output)