from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from paths import PathName, generate_path, nearest_point, normalize_angle
from vehicle_dynamics import DynamicBicycleModel, VehicleParams, VehicleState


@dataclass
class StanleyConfig:
    k_cte: float = 0.9
    k_soft: float = 1.0
    kd_yaw_rate: float = 0.15
    max_steer_rate: float = np.deg2rad(120.0)
    road_half_width: float = 5.0
    max_steps: int = 1200


class StanleyPIDController:
    def __init__(self, config: StanleyConfig | None = None, params: VehicleParams | None = None):
        self.config = config or StanleyConfig()
        self.params = params or VehicleParams()
        self.last_steer = 0.0

    def reset(self):
        self.last_steer = 0.0

    def control(self, lateral_error: float, heading_error: float, yaw_rate: float, speed: float) -> float:
        cfg = self.config
        steer = -heading_error - np.arctan2(cfg.k_cte * lateral_error, cfg.k_soft + speed) - cfg.kd_yaw_rate * yaw_rate
        max_delta = cfg.max_steer_rate * self.params.dt
        steer = float(np.clip(steer, self.last_steer - max_delta, self.last_steer + max_delta))
        steer = float(np.clip(steer, -self.params.max_steer, self.params.max_steer))
        self.last_steer = steer
        return steer


def simulate_stanley(
    path_type: PathName,
    target_speed: float = 12.0,
    mu: float = 1.0,
    side_wind_force: float = 0.0,
    speed_scale: float = 1.0,
    config: StanleyConfig | None = None,
) -> dict[str, np.ndarray | float | str]:
    path = generate_path(path_type, target_speed=target_speed)
    params = VehicleParams()
    model = DynamicBicycleModel(params)
    controller = StanleyPIDController(config=config, params=params)

    yaw0 = float(path.yaw[0])
    model.reset(VehicleState(x=float(path.x[0]), y=float(path.y[0]), yaw=yaw0, vx=float(path.speed[0] * speed_scale)))
    controller.reset()

    history: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "yaw": [],
        "vx": [],
        "lateral_error": [],
        "heading_error": [],
        "steer": [],
        "decision_time": [],
    }
    cfg = config or StanleyConfig()
    for _ in range(cfg.max_steps):
        s = model.state
        idx, lateral_error, path_yaw, _, ref_speed = nearest_point(path, s.x, s.y)
        heading_error = float(normalize_angle(s.yaw - path_yaw))
        t0 = time.perf_counter()
        steer = controller.control(lateral_error, heading_error, s.yaw_rate, s.vx)
        history["decision_time"].append(time.perf_counter() - t0)
        model.step(steer, target_speed=ref_speed * speed_scale, mu=mu, side_wind_force=side_wind_force)

        history["x"].append(s.x)
        history["y"].append(s.y)
        history["yaw"].append(s.yaw)
        history["vx"].append(s.vx)
        history["lateral_error"].append(lateral_error)
        history["heading_error"].append(heading_error)
        history["steer"].append(steer)

        if abs(lateral_error) > cfg.road_half_width:
            break
        if not path.closed and idx >= len(path.x) - 4:
            break

    result = {k: np.asarray(v, dtype=np.float64) for k, v in history.items()}
    result["path_name"] = path.name
    result["ref_x"] = path.x
    result["ref_y"] = path.y
    return result
