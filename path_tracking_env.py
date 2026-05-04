from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from paths import PathName, generate_path, nearest_point, normalize_angle
from vehicle_dynamics import DynamicBicycleModel, VehicleParams, VehicleState


@dataclass
class DisturbanceConfig:
    mu: float = 1.0
    side_wind_force: float = 0.0
    speed_scale: float = 1.0
    randomize: bool = False


class PathTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        path_type: PathName = "double_lane_change",
        max_episode_steps: int = 1200,
        target_speed: float = 12.0,
        road_half_width: float = 5.0,
        disturbance: DisturbanceConfig | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.path_type = path_type
        self.path_names = ("double_lane_change", "snake", "circle")
        self.max_episode_steps = max_episode_steps
        self.target_speed = target_speed
        self.road_half_width = road_half_width
        self.disturbance = disturbance or DisturbanceConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.params = VehicleParams()
        self.model = DynamicBicycleModel(self.params)
        self.max_steer_rate = np.deg2rad(120.0)
        self.prev_idx = 0
        self.step_count = 0
        self.last_steer = 0.0
        self.path = generate_path("double_lane_change", target_speed=target_speed)

        high = np.array([2.0, 1.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.params.max_steer], dtype=np.float32),
            high=np.array([self.params.max_steer], dtype=np.float32),
            dtype=np.float32,
        )
        self._fig = None
        self._ax = None

    def _select_path(self):
        if self.path_type == "mixed":
            name = self.rng.choice(self.path_names)
        else:
            name = self.path_type
        self.path = generate_path(name, target_speed=self.target_speed)

    def _current_disturbance(self) -> DisturbanceConfig:
        if not self.disturbance.randomize:
            return self.disturbance
        return DisturbanceConfig(
            mu=float(self.rng.uniform(0.75, 1.0)),
            side_wind_force=float(self.rng.uniform(-180.0, 180.0)),
            speed_scale=float(self.rng.uniform(0.85, 1.15)),
            randomize=True,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._select_path()
        self.active_disturbance = self._current_disturbance()

        start_idx = 0
        start_yaw = float(self.path.yaw[start_idx])
        normal = np.array([-np.sin(start_yaw), np.cos(start_yaw)])
        offset = float(self.rng.normal(0.0, 0.25))
        x0 = float(self.path.x[start_idx] + offset * normal[0])
        y0 = float(self.path.y[start_idx] + offset * normal[1])
        v0 = float(self.path.speed[start_idx] * self.active_disturbance.speed_scale)
        yaw0 = float(normalize_angle(start_yaw + self.rng.normal(0.0, 0.03)))
        self.model.reset(VehicleState(x=x0, y=y0, yaw=yaw0, vx=v0))
        self.prev_idx = start_idx
        self.step_count = 0
        self.last_steer = 0.0
        obs, info = self._get_obs_info()
        return obs, info

    def step(self, action):
        desired_steer = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        max_delta = self.max_steer_rate * self.params.dt
        steer = float(np.clip(desired_steer, self.last_steer - max_delta, self.last_steer + max_delta))
        steer = float(np.clip(steer, -self.params.max_steer, self.params.max_steer))
        steer_rate = (steer - self.last_steer) / self.params.dt

        _, _, _, _, ref_speed = nearest_point(self.path, self.model.state.x, self.model.state.y)
        target_speed = ref_speed * self.active_disturbance.speed_scale
        self.model.step(
            steer,
            target_speed=target_speed,
            mu=self.active_disturbance.mu,
            side_wind_force=self.active_disturbance.side_wind_force,
        )
        self.last_steer = steer
        self.step_count += 1

        obs, info = self._get_obs_info()
        progress = self._progress_reward(info["nearest_idx"])
        e_y = abs(info["lateral_error"])
        e_psi = abs(info["heading_error"])
        reward = (
            0.8 * progress
            - 1.2 * e_y
            - 0.35 * e_psi
            - 0.06 * abs(self.model.state.yaw_rate)
            - 0.015 * abs(steer_rate)
            - 0.04 * abs(steer)
        )

        terminated = bool(e_y > self.road_half_width or abs(e_psi) > 1.8)
        reached_end = (not self.path.closed) and info["nearest_idx"] >= len(self.path.x) - 4
        truncated = bool(self.step_count >= self.max_episode_steps or reached_end)
        if terminated:
            reward -= 30.0
        if reached_end and not terminated:
            reward += 30.0
        info.update({"steer": steer, "steer_rate": steer_rate, "reward": reward})
        return obs, float(reward), terminated, truncated, info

    def _progress_reward(self, idx: int) -> float:
        if self.path.closed:
            diff = idx - self.prev_idx
            if diff < -0.5 * len(self.path.x):
                diff += len(self.path.x)
            if diff > 0.5 * len(self.path.x):
                diff -= len(self.path.x)
            self.prev_idx = idx
            return float(np.clip(diff / 3.0, -1.0, 1.0))
        old_s = self.path.s[self.prev_idx]
        new_s = self.path.s[idx]
        self.prev_idx = max(self.prev_idx, idx)
        return float(np.clip((new_s - old_s) / max(self.target_speed * self.params.dt, 1e-3), -1.0, 1.0))

    def _get_obs_info(self):
        s = self.model.state
        idx, lateral_error, path_yaw, curvature, ref_speed = nearest_point(self.path, s.x, s.y)
        heading_error = float(normalize_angle(s.yaw - path_yaw))
        progress = float(self.path.s[idx] / max(self.path.length, 1e-6))
        obs = np.array(
            [
                np.clip(lateral_error / self.road_half_width, -2.0, 2.0),
                np.clip(heading_error / np.pi, -1.0, 1.0),
                np.clip(s.vy / 5.0, -3.0, 3.0),
                np.clip(s.yaw_rate / 2.0, -3.0, 3.0),
                np.clip(s.vx / max(self.target_speed, 1e-6), 0.0, 2.0),
                np.clip(curvature * 20.0, -2.0, 2.0),
                np.clip(self.last_steer / self.params.max_steer, -1.0, 1.0),
                np.clip(progress, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        info = {
            "x": s.x,
            "y": s.y,
            "yaw": s.yaw,
            "vx": s.vx,
            "vy": s.vy,
            "yaw_rate": s.yaw_rate,
            "nearest_idx": idx,
            "lateral_error": lateral_error,
            "heading_error": heading_error,
            "curvature": curvature,
            "ref_speed": ref_speed,
            "path_name": self.path.name,
        }
        return obs, info

    def render(self):
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 4.5))
        self._ax.clear()
        self._ax.plot(self.path.x, self.path.y, "k--", linewidth=1.5)
        self._ax.scatter([self.model.state.x], [self.model.state.y], c="tab:red", s=28)
        self._ax.axis("equal")
        self._ax.grid(True)
        self._fig.canvas.draw()
        if self.render_mode == "rgb_array":
            w, h = self._fig.canvas.get_width_height()
            return np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.pause(0.001)
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
