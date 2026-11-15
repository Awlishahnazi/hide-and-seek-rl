import argparse
import torch
import numpy as np
import json
import os
from hideseek.env.pz_env import HideSeekEnv
from hideseek.mappo.trainer import MAPPOTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=20, help="Number of training updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_hiders", type=int, default=2)
    parser.add_argument("--n_seekers", type=int, default=2)
    parser.add_argument("--n_boxes", type=int, default=2)
    parser.add_argument("--vision_radius", type=int, default=5, help="Vision radius for agents")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment and trainer
    env = HideSeekEnv(
        map_size=15,
        n_hiders=args.n_hiders,
        n_seekers=args.n_seekers,
        n_boxes=args.n_boxes,
        vision_radius=args.vision_radius,
        max_steps=args.max_steps
    )
    trainer = MAPPOTrainer(env)

    print(f"Starting training: {args.updates} updates | Team {args.n_hiders}v{args.n_seekers} | Vision {args.vision_radius} | Boxes {args.n_boxes}")
    
    def save_callback(update, trainer):
        if (update + 1) % 100 == 0 or update == args.updates - 1:
            prefix = f"h{args.n_hiders}_s{args.n_seekers}_v{args.vision_radius}_b{args.n_boxes}_seed{args.seed}_update{update+1}"
            os.makedirs("results/checkpoints", exist_ok=True)
            torch.save(trainer.actors["hider"].state_dict(), f"results/checkpoints/hider_actor_{prefix}.pth")
            torch.save(trainer.actors["seeker"].state_dict(), f"results/checkpoints/seeker_actor_{prefix}.pth")
            print(f"[Checkpoint] Saved models at update {update+1}")

    # Run training with callback
    trainer.train(total_updates=args.updates, save_callback=save_callback)
    
    print("Training complete! Model weights saved.")
    
    # Save the final models (also handled by the callback for the last update, but keeping this for safety)
    os.makedirs("results/checkpoints", exist_ok=True)
    prefix = f"h{args.n_hiders}_s{args.n_seekers}_v{args.vision_radius}_b{args.n_boxes}_seed{args.seed}"
    hider_path = f"results/checkpoints/hider_actor_{prefix}.pth"
    seeker_path = f"results/checkpoints/seeker_actor_{prefix}.pth"
    
    torch.save(trainer.actors["hider"].state_dict(), hider_path)
    torch.save(trainer.actors["seeker"].state_dict(), seeker_path)
    
    print(f"Final Checkpoints saved to:\n- {hider_path}\n- {seeker_path}")

if __name__ == "__main__":
    main()