from __future__ import annotations

import argparse
import csv
from pathlib import Path

from path_tracking_env import DisturbanceConfig, PathTrackingEnv
from plot_results import ensure_dir, plot_training_curves


class EpisodeErrorCallback:
    def __init__(self, log_dir: str | Path):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, path: Path):
                super().__init__()
                self.path = path
                self.episode = 0
                self.errors: list[float] = []
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "mean_abs_error"])

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for info, done in zip(infos, dones):
                    if "lateral_error" in info:
                        self.errors.append(abs(float(info["lateral_error"])))
                    if done:
                        mean_error = sum(self.errors) / max(len(self.errors), 1)
                        with self.path.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([self.episode, mean_error])
                        self.episode += 1
                        self.errors = []
                return True

        self.callback = _Callback(Path(log_dir) / "training_metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAC path tracking controller.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--path", choices=["double_lane_change", "snake", "circle", "mixed"], default="mixed")
    parser.add_argument("--target-speed", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", default="models/sac_path_tracking.zip")
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny training job for wiring checks.")
    return parser.parse_args()


def main() -> None:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3/torch is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    if args.smoke_test:
        args.timesteps = min(args.timesteps, 2_000)
        args.learning_starts = min(args.learning_starts, 100)
        args.buffer_size = min(args.buffer_size, 10_000)

    ensure_dir(Path(args.model_path).parent)
    ensure_dir(args.log_dir)

    def make_env():
        env = PathTrackingEnv(
            path_type=args.path,
            target_speed=args.target_speed,
            disturbance=DisturbanceConfig(randomize=True),
            seed=args.seed,
        )
        return Monitor(env, filename=str(Path(args.log_dir) / "monitor"))

    env = DummyVecEnv([make_env])
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        verbose=1,
    )
    callback = EpisodeErrorCallback(args.log_dir).callback
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(args.model_path)
    plot_training_curves(args.log_dir, "results")
    print(f"Saved SAC model to {args.model_path}")


if __name__ == "__main__":
    main()
