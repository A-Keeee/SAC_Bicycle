from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from baseline_pid import simulate_stanley
from path_tracking_env import DisturbanceConfig, PathTrackingEnv
from plot_results import (
    compute_metrics,
    plot_error_comparison,
    plot_steering,
    plot_tracking,
    save_metrics_table,
)


SCENARIOS = {
    "nominal": DisturbanceConfig(mu=1.0, side_wind_force=0.0, speed_scale=1.0),
    "low_mu": DisturbanceConfig(mu=0.75, side_wind_force=0.0, speed_scale=1.0),
    "side_wind": DisturbanceConfig(mu=1.0, side_wind_force=180.0, speed_scale=1.0),
    "speed_change": DisturbanceConfig(mu=1.0, side_wind_force=0.0, speed_scale=1.15),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC and PID/Stanley path tracking controllers.")
    parser.add_argument("--model-path", default="models/sac_path_tracking.zip")
    parser.add_argument("--paths", nargs="+", default=["double_lane_change", "snake", "circle"])
    parser.add_argument("--scenario", choices=["nominal", "low_mu", "side_wind", "speed_change", "all"], default="all")
    parser.add_argument("--target-speed", type=float, default=12.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def rollout_sac(model, path_type: str, disturbance: DisturbanceConfig, target_speed: float, seed: int) -> dict:
    env = PathTrackingEnv(path_type=path_type, disturbance=disturbance, target_speed=target_speed, seed=seed)
    obs, info = env.reset(seed=seed)
    history = {
        "x": [],
        "y": [],
        "yaw": [],
        "vx": [],
        "lateral_error": [],
        "heading_error": [],
        "steer": [],
        "decision_time": [],
    }
    while True:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        history["decision_time"].append(time.perf_counter() - t0)
        obs, _, terminated, truncated, info = env.step(action)
        history["x"].append(info["x"])
        history["y"].append(info["y"])
        history["yaw"].append(info["yaw"])
        history["vx"].append(info["vx"])
        history["lateral_error"].append(info["lateral_error"])
        history["heading_error"].append(info["heading_error"])
        history["steer"].append(info["steer"])
        if terminated or truncated:
            break
    result = {k: np.asarray(v, dtype=np.float64) for k, v in history.items()}
    result["path_name"] = env.path.name
    result["ref_x"] = env.path.x
    result["ref_y"] = env.path.y
    env.close()
    return result


def main() -> None:
    args = parse_args()
    scenario_names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    model = None
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise SystemExit(
                "stable-baselines3/torch is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from exc
        model = SAC.load(model_path)
    else:
        print(f"SAC model not found at {model_path}; generating PID/Stanley results only.")

    rows = []
    for scenario_name in scenario_names:
        disturbance = SCENARIOS[scenario_name]
        for path_type in args.paths:
            pid_result = simulate_stanley(
                path_type=path_type,
                target_speed=args.target_speed,
                mu=disturbance.mu,
                side_wind_force=disturbance.side_wind_force,
                speed_scale=disturbance.speed_scale,
            )
            rows.append(compute_metrics(pid_result, "PID_Stanley", scenario_name))
            plot_tracking(pid_result, "PID_Stanley", args.output_dir)
            plot_steering(pid_result, "PID_Stanley", args.output_dir)

            sac_result = None
            if model is not None:
                sac_result = rollout_sac(model, path_type, disturbance, args.target_speed, args.seed)
                rows.append(compute_metrics(sac_result, "SAC", scenario_name))
                plot_tracking(sac_result, "SAC", args.output_dir)
                plot_steering(sac_result, "SAC", args.output_dir)

            plot_error_comparison(sac_result, pid_result, args.output_dir)

    save_metrics_table(rows, args.output_dir)
    print(f"Saved evaluation figures and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
