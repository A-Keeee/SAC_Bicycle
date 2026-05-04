from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baseline_pid import StanleyPIDController
from path_tracking_env import DisturbanceConfig, PathTrackingEnv


SCENARIOS = {
    "nominal": DisturbanceConfig(mu=1.0, side_wind_force=0.0, speed_scale=1.0),
    "low_mu": DisturbanceConfig(mu=0.75, side_wind_force=0.0, speed_scale=1.0),
    "side_wind": DisturbanceConfig(mu=1.0, side_wind_force=180.0, speed_scale=1.0),
    "speed_change": DisturbanceConfig(mu=1.0, side_wind_force=0.0, speed_scale=1.15),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dynamic path tracking simulation.")
    parser.add_argument("--controller", choices=["sac", "pid"], default="sac")
    parser.add_argument("--model-path", default="models/sac_path_tracking.zip")
    parser.add_argument("--path", choices=["double_lane_change", "snake", "circle"], default="double_lane_change")
    parser.add_argument("--compare", action="store_true", help="Show SAC and PID/Stanley in the same animation.")
    parser.add_argument("--all-paths", action="store_true", help="Show double lane change, snake, and circle together.")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="nominal")
    parser.add_argument("--target-speed", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--interval", type=int, default=35, help="Animation interval in milliseconds.")
    parser.add_argument("--save", default="", help="Optional output file, e.g. results/sac_snake.gif or .mp4")
    parser.add_argument("--no-display", action="store_true", help="Save animation without opening an interactive window.")
    return parser.parse_args()


def load_sac_model(model_path: str):
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3/torch is not installed. Run: python -m pip install -r requirements.txt") from exc

    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"SAC model not found: {path}. Train first with train_sac.py or use --controller pid.")
    return SAC.load(path)


def vehicle_outline(x: float, y: float, yaw: float) -> np.ndarray:
    length = 4.6
    width = 1.9
    rear_to_center = 1.6
    corners = np.array(
        [
            [length - rear_to_center, width / 2.0],
            [length - rear_to_center, -width / 2.0],
            [-rear_to_center, -width / 2.0],
            [-rear_to_center, width / 2.0],
        ]
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return corners @ rot.T + np.array([x, y])


def save_or_show(animation, fig, args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if args.save:
        output = Path(args.save)
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        if suffix == ".gif":
            animation.save(output, writer=PillowWriter(fps=max(1, int(1000 / args.interval))))
        elif suffix == ".mp4":
            animation.save(output, writer=FFMpegWriter(fps=max(1, int(1000 / args.interval))))
        else:
            raise SystemExit("Unsupported animation format. Use .gif or .mp4")
        print(f"Saved dynamic simulation to {output}")

    if not args.no_display:
        plt.show()
    plt.close(fig)


def make_agent(path_name: str, controller: str, args, model):
    env = PathTrackingEnv(
        path_type=path_name,
        target_speed=args.target_speed,
        disturbance=SCENARIOS[args.scenario],
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)
    agent = {
        "controller": controller,
        "env": env,
        "obs": obs,
        "info": info,
        "pid": StanleyPIDController(params=env.params) if controller == "pid" else None,
        "model": model if controller == "sac" else None,
        "done": False,
        "reason": "running",
        "x": [info["x"]],
        "y": [info["y"]],
        "error": [info["lateral_error"]],
        "steer": [0.0],
        "speed": [info["vx"]],
    }
    return agent


def step_agent(agent) -> None:
    if agent["done"]:
        return
    env = agent["env"]
    info = agent["info"]
    if agent["controller"] == "sac":
        action, _ = agent["model"].predict(agent["obs"], deterministic=True)
    else:
        steer = agent["pid"].control(
            info["lateral_error"],
            info["heading_error"],
            info["yaw_rate"],
            info["vx"],
        )
        action = np.array([steer], dtype=np.float32)

    obs, _, terminated, truncated, info = env.step(action)
    agent["obs"] = obs
    agent["info"] = info
    agent["x"].append(info["x"])
    agent["y"].append(info["y"])
    agent["error"].append(info["lateral_error"])
    agent["steer"].append(info["steer"])
    agent["speed"].append(info["vx"])
    if terminated or truncated:
        agent["done"] = True
        agent["reason"] = "terminated" if terminated else "finished"


def run_compare(args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon

    path_names = ["double_lane_change", "snake", "circle"] if args.all_paths else [args.path]
    model = load_sac_model(args.model_path)
    rows = len(path_names)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.2 * rows), squeeze=False)
    fig.suptitle(f"SAC vs PID/Stanley dynamic simulation | scenario={args.scenario}")

    panels = []
    colors = {"sac": "tab:blue", "pid": "tab:orange"}
    labels = {"sac": "SAC", "pid": "PID/Stanley"}
    for row, path_name in enumerate(path_names):
        sac_agent = make_agent(path_name, "sac", args, model)
        pid_agent = make_agent(path_name, "pid", args, model)
        agents = [sac_agent, pid_agent]
        env = sac_agent["env"]

        ax_map = axes[row][0]
        ax_err = axes[row][1]
        ax_map.plot(env.path.x, env.path.y, "k--", linewidth=1.3, label="reference")
        pad = 8.0
        ax_map.set_xlim(float(np.min(env.path.x) - pad), float(np.max(env.path.x) + pad))
        ax_map.set_ylim(float(np.min(env.path.y) - pad), float(np.max(env.path.y) + pad))
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_title(path_name)
        ax_map.set_xlabel("x (m)")
        ax_map.set_ylabel("y (m)")
        ax_map.grid(True)

        ax_err.axhline(env.road_half_width, color="tab:red", linestyle="--", linewidth=1)
        ax_err.axhline(-env.road_half_width, color="tab:red", linestyle="--", linewidth=1)
        ax_err.set_title(f"{path_name} lateral error")
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("error (m)")
        ax_err.grid(True)

        panel = {"agents": agents, "ax_map": ax_map, "ax_err": ax_err, "artists": []}
        for agent in agents:
            color = colors[agent["controller"]]
            (traj_line,) = ax_map.plot([], [], color=color, linewidth=2.0, label=labels[agent["controller"]])
            patch = Polygon(
                vehicle_outline(agent["info"]["x"], agent["info"]["y"], agent["info"]["yaw"]),
                closed=True,
                color=color,
                alpha=0.78,
            )
            ax_map.add_patch(patch)
            (err_line,) = ax_err.plot([], [], color=color, linewidth=1.8, label=labels[agent["controller"]])
            panel["artists"].append({"agent": agent, "traj": traj_line, "patch": patch, "err": err_line})
        ax_map.legend(loc="best")
        ax_err.legend(loc="best")
        panels.append(panel)

    def update(frame: int):
        updated = []
        for panel in panels:
            max_error = 1.0
            max_steps = 100
            for item in panel["artists"]:
                agent = item["agent"]
                step_agent(agent)
                info = agent["info"]
                steps = np.arange(len(agent["x"]))
                item["traj"].set_data(agent["x"], agent["y"])
                item["patch"].set_xy(vehicle_outline(info["x"], info["y"], info["yaw"]))
                item["err"].set_data(steps, agent["error"])
                max_error = max(max_error, float(np.max(np.abs(agent["error"]))) + 0.5)
                max_steps = max(max_steps, len(steps))
                updated.extend([item["traj"], item["patch"], item["err"]])
            panel["ax_err"].set_xlim(0, max_steps)
            y_abs = max(panel["artists"][0]["agent"]["env"].road_half_width * 1.1, max_error)
            panel["ax_err"].set_ylim(-y_abs, y_abs)
        return updated

    animation = FuncAnimation(fig, update, frames=args.max_steps, interval=args.interval, blit=False, repeat=False)
    fig.tight_layout()
    save_or_show(animation, fig, args)
    for panel in panels:
        for item in panel["artists"]:
            item["agent"]["env"].close()


def main() -> None:
    args = parse_args()
    if args.no_display:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon

    if args.compare or args.all_paths:
        run_compare(args)
        return

    env = PathTrackingEnv(
        path_type=args.path,
        target_speed=args.target_speed,
        disturbance=SCENARIOS[args.scenario],
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)
    model = load_sac_model(args.model_path) if args.controller == "sac" else None
    pid = StanleyPIDController(params=env.params) if args.controller == "pid" else None

    history = {
        "x": [info["x"]],
        "y": [info["y"]],
        "error": [info["lateral_error"]],
        "steer": [0.0],
        "speed": [info["vx"]],
        "reward": [0.0],
    }
    finished = {"done": False, "reason": "running"}

    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.0, 1.0])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_error = fig.add_subplot(grid[0, 1])
    ax_steer = fig.add_subplot(grid[1, 1])
    ax_speed = fig.add_subplot(grid[2, 1])

    ax_map.plot(env.path.x, env.path.y, "k--", linewidth=1.5, label="reference")
    (traj_line,) = ax_map.plot([], [], color="tab:blue", linewidth=2.0, label=args.controller.upper())
    vehicle_patch = Polygon(vehicle_outline(info["x"], info["y"], info["yaw"]), closed=True, color="tab:red", alpha=0.85)
    ax_map.add_patch(vehicle_patch)
    ax_map.axis("equal")
    pad = 8.0
    ax_map.set_xlim(float(np.min(env.path.x) - pad), float(np.max(env.path.x) + pad))
    ax_map.set_ylim(float(np.min(env.path.y) - pad), float(np.max(env.path.y) + pad))
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.grid(True)
    ax_map.legend(loc="best")

    (error_line,) = ax_error.plot([], [], color="tab:orange")
    ax_error.axhline(env.road_half_width, color="tab:red", linestyle="--", linewidth=1)
    ax_error.axhline(-env.road_half_width, color="tab:red", linestyle="--", linewidth=1)
    ax_error.set_ylabel("error (m)")
    ax_error.grid(True)

    (steer_line,) = ax_steer.plot([], [], color="tab:green")
    ax_steer.set_ylabel("steer (deg)")
    ax_steer.grid(True)

    (speed_line,) = ax_speed.plot([], [], color="tab:purple")
    ax_speed.set_xlabel("step")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.grid(True)

    title = fig.suptitle("")

    def choose_action(current_obs, current_info):
        if args.controller == "sac":
            action, _ = model.predict(current_obs, deterministic=True)
            return action
        steer = pid.control(
            current_info["lateral_error"],
            current_info["heading_error"],
            current_info["yaw_rate"],
            current_info["vx"],
        )
        return np.array([steer], dtype=np.float32)

    def update(frame: int):
        nonlocal obs, info
        if not finished["done"]:
            action = choose_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            history["x"].append(info["x"])
            history["y"].append(info["y"])
            history["error"].append(info["lateral_error"])
            history["steer"].append(info["steer"])
            history["speed"].append(info["vx"])
            history["reward"].append(reward)
            if terminated or truncated:
                finished["done"] = True
                finished["reason"] = "terminated" if terminated else "finished"

        steps = np.arange(len(history["x"]))
        traj_line.set_data(history["x"], history["y"])
        vehicle_patch.set_xy(vehicle_outline(info["x"], info["y"], info["yaw"]))
        error_line.set_data(steps, history["error"])
        steer_line.set_data(steps, np.rad2deg(history["steer"]))
        speed_line.set_data(steps, history["speed"])

        right = max(100, len(steps))
        ax_error.set_xlim(0, right)
        ax_steer.set_xlim(0, right)
        ax_speed.set_xlim(0, right)
        ax_error.set_ylim(-max(env.road_half_width * 1.1, np.max(np.abs(history["error"])) + 0.5), max(env.road_half_width * 1.1, np.max(np.abs(history["error"])) + 0.5))
        ax_steer.set_ylim(-35, 35)
        ax_speed.set_ylim(0, max(args.target_speed * 1.5, np.max(history["speed"]) + 2.0))
        title.set_text(
            f"{args.controller.upper()} | {env.path.name} | {args.scenario} | "
            f"step={len(steps)-1} | e_y={info['lateral_error']:.2f} m | "
            f"delta={np.rad2deg(info.get('steer', 0.0)):.1f} deg | {finished['reason']}"
        )
        return traj_line, vehicle_patch, error_line, steer_line, speed_line, title

    animation = FuncAnimation(fig, update, frames=args.max_steps, interval=args.interval, blit=False, repeat=False)
    fig.tight_layout()

    save_or_show(animation, fig, args)
    env.close()


if __name__ == "__main__":
    main()
