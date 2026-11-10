import imageio
import numpy as np
import os
from hideseek.env.core import HideSeekCore
from hideseek.env.render import PygameRenderer

def generate_random_episode_gif(filename="assets/random_episode.gif"):
    # Initialize environment and renderer
    core = HideSeekCore(seed=42, map_size=15)
    renderer = PygameRenderer(cell_size=40)
    
    frames = []
    
    # Run for 120 steps (to capture prep phase and chase phase)
    for step in range(120):
        # Generate random actions for all agents
        actions = {agent: np.random.randint(0, 5) for agent in core.agent_ids}
        
        # Step the environment
        rewards, done = core.step(actions)
        
        # Render the frame
        frame = renderer.render_frame(core)
        frames.append(frame)
        
        if done:
            break
            
    renderer.close()
    
    # Ensure assets folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save as GIF
    print(f"Saving GIF to {filename}...")
    imageio.mimsave(filename, frames, fps=10)
    print("Done! Check assets/random_episode.gif")

if __name__ == "__main__":
    generate_random_episode_gif()