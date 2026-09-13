"""Generate a .pruned_init file for a bddl task by sampling valid resets.

Each reset randomizes object placement within the regions declared in the
bddl file; `OffScreenRenderEnv.reset()` already retries internally on
`RandomizationError`, so every captured state is a valid, non-intersecting
placement. This mirrors the format of the existing .pruned_init files: a
(num_states, state_dim) float64 numpy array of flattened mujoco sim states,
torch.save'd.

Usage:
    python scripts/generate_init_states.py \
        --bddl_file path/to/task.bddl \
        --out path/to/task.pruned_init \
        --num_states 50
"""
import argparse
import os

import numpy as np
import torch

from libero.libero.envs import OffScreenRenderEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl_file", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = OffScreenRenderEnv(bddl_file_name=args.bddl_file)
    env.seed(args.seed)

    states = []
    for _ in range(args.num_states):
        env.reset()
        states.append(env.get_sim_state())
    env.close()

    states = np.stack(states).astype(np.float64)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(states, args.out)
    print(f"Saved {states.shape[0]} init states (dim={states.shape[1]}) to {args.out}")


if __name__ == "__main__":
    main()
