"""Evaluate a lerobot policy on one or more LIBERO-Constrained suites, and
report BOTH task success and constraint-violation as independent results
per episode -- matching what scripts/run_libero_step.py's STeP pipeline now
reports via execution_summary.json's "constraint_check" field.

This can't be done with the plain `lerobot-eval` CLI: constraint checking
needs the live env instance's logged trajectory before it gets closed, and
the CLI never surfaces that. So this script drives lerobot's own
eval_policy() directly (same function lerobot-eval itself calls), then reads
env.envs[0]._env.check_constraints() right after each rollout and before
closing.

Usage:
    python scripts/eval_policy_with_constraints.py \
        --policy_path HuggingFaceVLA/smolvla_libero \
        --suites libero_spatial_constrained libero_object_constrained libero_goal_constrained \
        --outdir eval_results/smolvla_all_constrained \
        --n_episodes 1 --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lerobot_src", default=None, help="Path to lerobot's src/ dir, if not already importable.")
    args = parser.parse_args()

    if args.lerobot_src:
        sys.path.insert(0, args.lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    from libero.libero import benchmark

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = str(device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        n_tasks = suite.get_num_tasks()

        for task_id in range(n_tasks):
            task_name = suite.get_task(task_id).name
            env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id])

            envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
            vec_env = envs[suite_name][task_id]

            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            policy.eval()
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=args.policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

            task_outdir = outdir / suite_name / f"{task_id}_{task_name}"
            task_outdir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                info = eval_policy(
                    env=vec_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=args.n_episodes,
                    max_episodes_rendered=args.n_episodes,
                    videos_dir=task_outdir,
                )

            constraint_result = None
            try:
                constraint_result = vec_env.envs[0]._env.check_constraints()
            except Exception as exc:
                constraint_result = {"error": str(exc)}

            vec_env.close()

            result = {
                "suite": suite_name,
                "task_id": task_id,
                "task_name": task_name,
                "pc_success": info["aggregated"]["pc_success"],
                "constraint_check": constraint_result,
            }
            all_results.append(result)
            print(f"[{suite_name}] task {task_id} ({task_name}): "
                  f"success={result['pc_success']}% | constraints={constraint_result}")

            with open(outdir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved full results to {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
