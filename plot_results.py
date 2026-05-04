from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def compute_metrics(result: dict, controller: str, scenario: str) -> dict[str, float | str]:
    e = np.asarray(result.get("lateral_error", []), dtype=np.float64)
    steer = np.asarray(result.get("steer", []), dtype=np.float64)
    decision_time = np.asarray(result.get("decision_time", []), dtype=np.float64)
    steer_rate = np.diff(steer) / 0.05 if len(steer) > 1 else np.array([0.0])
    return {
        "controller": controller,
        "scenario": scenario,
        "path": str(result.get("path_name", scenario)),
        "rmse_m": float(np.sqrt(np.mean(e * e))) if len(e) else np.nan,
        "max_abs_error_m": float(np.max(np.abs(e))) if len(e) else np.nan,
        "steer_rate_std_rad_s": float(np.std(steer_rate)) if len(steer_rate) else np.nan,
        "avg_decision_time_ms": float(np.mean(decision_time) * 1000.0) if len(decision_time) else np.nan,
        "steps": int(len(e)),
    }


def save_metrics_table(rows: list[dict], output_dir: str | Path = "results") -> None:
    output = ensure_dir(output_dir)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    csv_path = output / "performance_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output / "performance_metrics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            values = []
            for name in fieldnames:
                value = row[name]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            f.write("| " + " | ".join(values) + " |\n")


def plot_training_curves(log_dir: str | Path = "results/logs", output_dir: str | Path = "results") -> None:
    log_dir = Path(log_dir)
    output = ensure_dir(output_dir)
    monitor_files = sorted(log_dir.glob("*.monitor.csv"))
    if monitor_files:
        rewards = []
        for file in monitor_files:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or line.startswith("r,"):
                        continue
                    parts = line.strip().split(",")
                    if len(parts) >= 1:
                        rewards.append(float(parts[0]))
        if rewards:
            plt.figure(figsize=(8, 4))
            plt.plot(rewards, label="episode reward")
            if len(rewards) >= 10:
                kernel = np.ones(10) / 10.0
                plt.plot(np.convolve(rewards, kernel, mode="valid"), label="moving avg")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / "training_reward_curve.png", dpi=160)
            plt.close()

    metrics_file = log_dir / "training_metrics.csv"
    if metrics_file.exists():
        data = np.genfromtxt(metrics_file, delimiter=",", names=True)
        if data.size:
            plt.figure(figsize=(8, 4))
            plt.plot(np.atleast_1d(data["episode"]), np.atleast_1d(data["mean_abs_error"]), label="mean abs lateral error")
            plt.xlabel("Episode")
            plt.ylabel("Error (m)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / "training_error_curve.png", dpi=160)
            plt.close()


def plot_tracking(result: dict, controller: str, output_dir: str | Path = "results") -> None:
    output = ensure_dir(output_dir)
    path_name = result["path_name"]
    plt.figure(figsize=(8, 5))
    plt.plot(result["ref_x"], result["ref_y"], "k--", label="reference")
    plt.plot(result["x"], result["y"], label=controller)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"{controller} tracking - {path_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"trajectory_{controller}_{path_name}.png", dpi=160)
    plt.close()


def plot_error_comparison(rl_result: dict | None, pid_result: dict, output_dir: str | Path = "results") -> None:
    output = ensure_dir(output_dir)
    path_name = pid_result["path_name"]
    plt.figure(figsize=(8, 4))
    if rl_result is not None:
        plt.plot(rl_result["lateral_error"], label="SAC")
    plt.plot(pid_result["lateral_error"], label="PID/Stanley")
    plt.xlabel("Step")
    plt.ylabel("Lateral error (m)")
    plt.title(f"Lateral error comparison - {path_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"lateral_error_comparison_{path_name}.png", dpi=160)
    plt.close()


def plot_steering(result: dict, controller: str, output_dir: str | Path = "results") -> None:
    output = ensure_dir(output_dir)
    path_name = result["path_name"]
    plt.figure(figsize=(8, 4))
    plt.plot(np.rad2deg(result["steer"]))
    plt.xlabel("Step")
    plt.ylabel("Steering angle (deg)")
    plt.title(f"{controller} steering - {path_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / f"steering_{controller}_{path_name}.png", dpi=160)
    plt.close()
