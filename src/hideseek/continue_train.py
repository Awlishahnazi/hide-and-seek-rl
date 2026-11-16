import argparse
import torch
import numpy as np
import os
import time
from hideseek.env.pz_env import HideSeekEnv
from hideseek.mappo.trainer import MAPPOTrainer
from hideseek.mappo.networks import ActorNetwork, CriticNetwork

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=500, help="Number of additional updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_prefix", type=str, default="h2_s2_v5_b2_seed42_update1000", help="Prefix of the checkpoint to load")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning (smaller than initial)")
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = HideSeekEnv(
        map_size=15,
        n_hiders=2,
        n_seekers=2,
        n_boxes=2,
        vision_radius=5,
        max_steps=args.max_steps
    )
    
    trainer = MAPPOTrainer(env, lr=args.lr)

    hider_path = f"results/checkpoints/hider_actor_{args.model_prefix}.pth"
    seeker_path = f"results/checkpoints/seeker_actor_{args.model_prefix}.pth"
    
    if os.path.exists(hider_path) and os.path.exists(seeker_path):
        print(f"Loading models from {args.model_prefix}...")
        trainer.actors["hider"].load_state_dict(torch.load(hider_path))
        trainer.actors["seeker"].load_state_dict(torch.load(seeker_path))
        print("Models loaded successfully!")
    else:
        print(f"Error: Model files not found at {hider_path} or {seeker_path}.")
        print("Please check the prefix or train from scratch.")
        return

    print(f"Continuing training for {args.updates} updates with LR={args.lr}...")
    
    def save_callback(update, trainer):
        if (update + 1) % 100 == 0 or update == args.updates - 1:
            new_prefix = f"{args.model_prefix}_plus{update+1}"
            os.makedirs("results/checkpoints", exist_ok=True)
            torch.save(trainer.actors["hider"].state_dict(), f"results/checkpoints/hider_actor_{new_prefix}.pth")
            torch.save(trainer.actors["seeker"].state_dict(), f"results/checkpoints/seeker_actor_{new_prefix}.pth")
            print(f"[Checkpoint] Saved models at update {update+1} as {new_prefix}")

    trainer.train(total_updates=args.updates, save_callback=save_callback)

    final_prefix = f"{args.model_prefix}_plus{args.updates}"
    torch.save(trainer.actors["hider"].state_dict(), f"results/checkpoints/hider_actor_{final_prefix}.pth")
    torch.save(trainer.actors["seeker"].state_dict(), f"results/checkpoints/seeker_actor_{final_prefix}.pth")
    print(f"Training complete! Final models saved as {final_prefix}")

if __name__ == "__main__":
    main()