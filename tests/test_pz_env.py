import numpy as np
import pytest
from hideseek.env.pz_env import HideSeekEnv

def test_pettingzoo_api():
    env = HideSeekEnv()
    assert len(env.agents) == 4
    
    obs, infos = env.reset(seed=42)
    assert len(obs) == 4
    assert "hider_0" in obs
    assert "seeker_0" in obs
    
    # Check observation shapes
    assert obs["hider_0"]["grid"].shape == (5, 11, 11)
    assert obs["hider_0"]["vec"].shape == (4,)
    
    # Step
    actions = {agent: 0 for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    assert len(obs) == 4
    assert len(rewards) == 4
    assert all(isinstance(t, bool) for t in terms.values())