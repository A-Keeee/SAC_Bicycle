from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paths import normalize_angle


@dataclass
class VehicleParams:
    mass: float = 1500.0
    iz: float = 2250.0
    lf: float = 1.45
    lr: float = 1.45
    cf: float = 32000.0
    cr: float = 34000.0
    max_steer: float = np.deg2rad(30.0)
    dt: float = 0.05
    min_vx: float = 1.0
    speed_tau: float = 1.2
    gravity: float = 9.81


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 10.0
    vy: float = 0.0
    yaw_rate: float = 0.0


class DynamicBicycleModel:
    """Two degree-of-freedom lateral dynamic bicycle model with simple speed tracking."""

    def __init__(self, params: VehicleParams | None = None):
        self.params = params or VehicleParams()
        self.state = VehicleState()

    def reset(self, state: VehicleState) -> VehicleState:
        self.state = VehicleState(**state.__dict__)
        return self.state

    def step(
        self,
        steer: float,
        target_speed: float | None = None,
        mu: float = 1.0,
        side_wind_force: float = 0.0,
    ) -> VehicleState:
        p = self.params
        s = self.state
        delta = float(np.clip(steer, -p.max_steer, p.max_steer))
        vx_safe = max(abs(s.vx), p.min_vx)

        alpha_f = np.arctan2(s.vy + p.lf * s.yaw_rate, vx_safe) - delta
        alpha_r = np.arctan2(s.vy - p.lr * s.yaw_rate, vx_safe)
        fyf = -p.cf * alpha_f
        fyr = -p.cr * alpha_r

        # A light saturation keeps low-friction tests bounded without adding a full tire model.
        max_front = mu * p.mass * p.gravity * p.lr / (p.lf + p.lr)
        max_rear = mu * p.mass * p.gravity * p.lf / (p.lf + p.lr)
        fyf = float(np.clip(fyf, -max_front, max_front))
        fyr = float(np.clip(fyr, -max_rear, max_rear))

        vx_dot = 0.0
        if target_speed is not None:
            vx_dot = (float(target_speed) - s.vx) / p.speed_tau
        vy_dot = (fyf * np.cos(delta) + fyr + side_wind_force) / p.mass - s.vx * s.yaw_rate
        yaw_rate_dot = (p.lf * fyf * np.cos(delta) - p.lr * fyr) / p.iz
        x_dot = s.vx * np.cos(s.yaw) - s.vy * np.sin(s.yaw)
        y_dot = s.vx * np.sin(s.yaw) + s.vy * np.cos(s.yaw)

        s.x += x_dot * p.dt
        s.y += y_dot * p.dt
        s.yaw = float(normalize_angle(s.yaw + s.yaw_rate * p.dt))
        s.vx = max(p.min_vx, s.vx + vx_dot * p.dt)
        s.vy += vy_dot * p.dt
        s.yaw_rate += yaw_rate_dot * p.dt
        return s
