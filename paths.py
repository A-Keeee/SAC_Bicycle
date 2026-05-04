from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


PathName = Literal["double_lane_change", "snake", "circle", "mixed"]


@dataclass(frozen=True)
class PathData:
    name: str
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    s: np.ndarray
    speed: np.ndarray
    closed: bool = False

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack((self.x, self.y))

    @property
    def length(self) -> float:
        return float(self.s[-1])


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _build_path(name: str, x: np.ndarray, y: np.ndarray, speed: float | np.ndarray, closed: bool) -> PathData:
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    yaw = np.arctan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)

    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate(([0.0], np.cumsum(ds)))
    if np.isscalar(speed):
        speed_arr = np.full_like(x, float(speed), dtype=np.float64)
    else:
        speed_arr = np.asarray(speed, dtype=np.float64)
    return PathData(name=name, x=x, y=y, yaw=yaw, curvature=curvature, s=s, speed=speed_arr, closed=closed)


def generate_path(path_type: PathName, n_points: int = 900, target_speed: float = 12.0) -> PathData:
    if path_type == "mixed":
        raise ValueError("'mixed' is a training option, not a concrete path.")

    if path_type == "double_lane_change":
        x = np.linspace(0.0, 160.0, n_points)
        lane_width = 3.5
        smooth = 8.0
        y = 0.5 * lane_width * (np.tanh((x - 45.0) / smooth) - np.tanh((x - 105.0) / smooth))
        speed = target_speed + 1.0 * np.sin(2.0 * np.pi * x / x[-1])
        return _build_path(path_type, x, y, speed, closed=False)

    if path_type == "snake":
        x = np.linspace(0.0, 170.0, n_points)
        y = 4.0 * np.sin(2.0 * np.pi * x / 55.0)
        speed = target_speed + 0.8 * np.sin(2.0 * np.pi * x / 90.0)
        return _build_path(path_type, x, y, speed, closed=False)

    if path_type == "circle":
        radius = 32.0
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        speed = np.full_like(x, min(target_speed, 10.0), dtype=np.float64)
        return _build_path(path_type, x, y, speed, closed=True)

    raise ValueError(f"Unknown path type: {path_type}")


def nearest_point(path: PathData, x: float, y: float) -> tuple[int, float, float, float, float]:
    points = path.xy
    distances = np.sum((points - np.array([x, y])) ** 2, axis=1)
    idx = int(np.argmin(distances))
    tangent_yaw = float(path.yaw[idx])
    normal = np.array([-np.sin(tangent_yaw), np.cos(tangent_yaw)])
    error_vec = np.array([x - path.x[idx], y - path.y[idx]])
    lateral_error = float(np.dot(error_vec, normal))
    return idx, lateral_error, tangent_yaw, float(path.curvature[idx]), float(path.speed[idx])
