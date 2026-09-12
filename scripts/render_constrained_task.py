"""Render a constrained-variant task (bddl file with added keep-out cylinders)
at a given init state and save the agentview image.

Usage:
    python scripts/render_constrained_task.py \
        --bddl_file libero/libero/bddl_files/libero_spatial_constrained/<task>.bddl \
        --init_states_file libero/libero/init_files/libero_spatial_constrained/<task>.pruned_init \
        --init_state_idx 0 \
        --out images/constrained_spatial/<task>.png
"""
import argparse
import os

import cv2
import torch

from libero.libero.envs import OffScreenRenderEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl_file", type=str, required=True)
    parser.add_argument("--init_states_file", type=str, required=True)
    parser.add_argument("--init_state_idx", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--camera_size", type=int, default=512)
    args = parser.parse_args()

    init_states = torch.load(args.init_states_file)

    env_args = {
        "bddl_file_name": args.bddl_file,
        "camera_heights": args.camera_size,
        "camera_widths": args.camera_size,
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    obs = env.set_init_state(init_states[args.init_state_idx])
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img = obs["agentview_image"][::-1, :, ::-1]
    cv2.imwrite(args.out, img)
    env.close()
    print(f"Saved image to {args.out}")


if __name__ == "__main__":
    main()
