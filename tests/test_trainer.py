from hideseek.env.pz_env import HideSeekEnv
from hideseek.mappo.trainer import MAPPOTrainer

def test_trainer_smoke():
    env = HideSeekEnv()
    trainer = MAPPOTrainer(env)
    
    # Run just 1 update with very few steps to test if it crashes
    trainer.collect_rollout(num_steps=10)
    print("Rollout collected successfully.")
    
    # Run training for 1 update
    trainer.train(total_updates=1)
    print("Training step completed successfully.")
    