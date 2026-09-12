"""Render a brand-new bddl task straight from its :init predicates, with no
saved .pruned_init file needed. Useful while iterating on a new task before
you've generated/collected any init states for it.

Usage:
    python scripts/render_new_task.py --bddl_file path/to/task.bddl --out images/preview.png
"""
import argparse
import os

import cv2

from libero.libero.envs import OffScreenRenderEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl_file", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--camera_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env_args = {
        "bddl_file_name": args.bddl_file,
        "camera_heights": args.camera_size,
        "camera_widths": args.camera_size,
    }
    env = OffScreenRenderEnv(**env_args)
    if args.seed is not None:
        env.seed(args.seed)
    obs = env.reset()
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img = obs["agentview_image"][::-1, :, ::-1]
    cv2.imwrite(args.out, img)
    env.close()
    print(f"Saved image to {args.out}")


if __name__ == "__main__":
    main()
