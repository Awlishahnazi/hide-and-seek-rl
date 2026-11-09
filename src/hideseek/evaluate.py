import imageio
import numpy as np
from hideseek.env.core import HideSeekCore
from hideseek.env.render import PygameRenderer

def generate_random_episode_gif(filename="assets/random_episode.gif"):
    # Initialize environment and renderer
    core = HideSeekCore(seed=42)
    renderer = PygameRenderer(cell_size=40)
    
    frames = []
    
    # Run for 100 steps (to keep the GIF size reasonable)
    for step in range(100):
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
    
    # Save as GIF
    print(f"Saving GIF to {filename}...")
    imageio.mimsave(filename, frames, fps=10)
    print("Done!")

if __name__ == "__main__":
    generate_random_episode_gif()