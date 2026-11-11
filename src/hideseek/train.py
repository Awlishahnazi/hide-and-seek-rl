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
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment and trainer
    env = HideSeekEnv(
        map_size=15,
        n_hiders=args.n_hiders,
        n_seekers=args.n_seekers,
        n_boxes=args.n_boxes
    )
    trainer = MAPPOTrainer(env)

    print(f"Starting baseline training for {args.updates} updates...")
    
    # We will collect simple stats for the first learning curve
    # In a full implementation, this would be logged to WandB or TensorBoard
    history = []
    
    # Note: The current trainer.train() runs the updates internally.
    # To plot learning curves, we need to extract stats. 
    # For this baseline, we'll just run the training and print the losses.
    # (We will add proper logging in the next step)
    
    trainer.train(total_updates=args.updates)
    
    print("Training complete! Model weights saved.")
    
    # Save the models (simple PyTorch save)
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(trainer.actors["hider"].state_dict(), "results/checkpoints/hider_actor.pth")
    torch.save(trainer.actors["seeker"].state_dict(), "results/checkpoints/seeker_actor.pth")
    print("Checkpoints saved to results/checkpoints/")

if __name__ == "__main__":
    main()