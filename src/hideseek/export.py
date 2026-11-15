import json
import os
import argparse
import numpy as np
from hideseek.env.pz_env import HideSeekEnv

def export_episode(output_file="demo/replay.json", n_hiders=2, n_seekers=2, vision_radius=5):
    env = HideSeekEnv(n_hiders=n_hiders, n_seekers=n_seekers, vision_radius=vision_radius)
    
    obs, _ = env.reset(seed=42)
    
    replay_data = {
        "map_size": env.map_size,
        "initial_grid": env.core.grid.tolist(),
        "steps": []
    }
    
    for step in range(200):
        actions_dict = {}
        for agent in env.agents:
            # --- استفاده از سیاست تصادفی برای حرکت تضمینی ---
            actions_dict[agent] = np.random.randint(0, 6) 
            
        obs, rewards, terms, truncs, infos = env.step(actions_dict)
        
        step_data = {
            "step": step,
            "is_prep_phase": env.core.is_prep_phase,
            "hiders": [list(map(int, p)) for p in env.core.hider_positions],
            "seekers": [list(map(int, p)) for p in env.core.seeker_positions],
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
    parser.add_argument("--output", type=str, default="demo/replay.json")
    args = parser.parse_args()
    
    export_episode(args.output)