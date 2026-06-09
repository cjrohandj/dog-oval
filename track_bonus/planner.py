"""Starter high-level planner for the 200 m track bonus.

The evaluator builds the official compact 5D track observation defined in
`track_bonus/controller_interface.py`. The high-level planner maps it to the
local joystick command consumed by the HW1 Go2 locomotion policy:

    5D track observation -> [vx, vy, yaw_rate]

This file is intentionally small.  It is a weak baseline and an interface
example, not a solved full-lap controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0
    weights_path: str | None = None
    vx_limit_mps: float = 0.8
    vy_limit_mps: float = 0.25
    yaw_rate_limit_radps: float = 0.6
    track_length_m: float = 200.0
    turn_radius_m: float = 18.25
    half_width_m: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
            "weights_path": self.weights_path,
            "vx_limit_mps": self.vx_limit_mps,
            "vy_limit_mps": self.vy_limit_mps,
            "yaw_rate_limit_radps": self.yaw_rate_limit_radps,
            "track_length_m": self.track_length_m,
            "turn_radius_m": self.turn_radius_m,
            "half_width_m": self.half_width_m,
        }


def _tanh_mlp_forward(obs: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    hidden = np.asarray(obs, dtype=np.float32)
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        hidden = hidden @ weight + bias
        if idx + 1 != len(weights):
            hidden = np.tanh(hidden)
    return np.asarray(hidden, dtype=np.float32)


def _load_mlp_weights(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    payload = np.load(path)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    layer_idx = 0
    while True:
        weight_key = f"W{layer_idx}"
        bias_key = f"b{layer_idx}"
        if weight_key not in payload or bias_key not in payload:
            break
        weights.append(np.asarray(payload[weight_key], dtype=np.float32))
        biases.append(np.asarray(payload[bias_key], dtype=np.float32))
        layer_idx += 1

    if not weights:
        raise ValueError(
            f"Expected learned planner weights in {path} with keys W0, b0, ... WN, bN."
        )

    input_mean = np.asarray(payload["input_mean"], dtype=np.float32) if "input_mean" in payload else np.zeros(5, dtype=np.float32)
    input_std = np.asarray(payload["input_std"], dtype=np.float32) if "input_std" in payload else np.ones(5, dtype=np.float32)

    if input_mean.shape != (5,) or input_std.shape != (5,):
        raise ValueError(f"Expected input_mean/input_std shape (5,), got {input_mean.shape} and {input_std.shape}.")
    if weights[0].shape[0] != 5:
        raise ValueError(f"Expected first MLP layer to accept 5 inputs, got shape {weights[0].shape}.")
    if weights[-1].shape[-1] != 3 or biases[-1].shape != (3,):
        raise ValueError(
            f"Expected final MLP layer to produce 3 outputs, got weight {weights[-1].shape} and bias {biases[-1].shape}."
        )
    return weights, biases, input_mean, np.maximum(input_std, 1e-6)


class StarterTrackPlanner:
    """Conservative coordinate-to-command baseline.

    The policy is deliberately simple and conservative. Students should improve
    it by changing this controller, replacing it with an MLP, or training a
    higher-level policy that produces the same command vector.
    """

    def __init__(self, config: StarterPlannerConfig) -> None:
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._mlp_weights: list[np.ndarray] | None = None
        self._mlp_biases: list[np.ndarray] | None = None
        self._input_mean = np.zeros(5, dtype=np.float32)
        self._input_std = np.ones(5, dtype=np.float32)
        if config.planner_type == "starter_pd":
            return
        if config.planner_type != "learned_mlp":
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        if not config.weights_path:
            raise ValueError("learned_mlp planner requires weights_path in planner config.")
        weights, biases, input_mean, input_std = _load_mlp_weights(Path(config.weights_path))
        self._mlp_weights = weights
        self._mlp_biases = biases
        self._input_mean = input_mean
        self._input_std = input_std

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        config = StarterPlannerConfig.load(path)
        if config.weights_path:
            weights_path = Path(config.weights_path)
            if not weights_path.is_absolute():
                config = StarterPlannerConfig.from_dict(
                    {
                        **config.to_dict(),
                        "weights_path": str((path.parent / weights_path).resolve()),
                    }
                )
        return cls(config)

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        if self.config.planner_type == "learned_mlp":
            return self.command_from_mlp(obs)
        return self.command_from_observation(obs)

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)

    def command_from_mlp(self, obs: TrackControllerObservation) -> np.ndarray:
        if self._mlp_weights is None or self._mlp_biases is None:
            raise RuntimeError("learned_mlp planner is missing loaded weights.")
        normalized = (obs.as_array() - self._input_mean) / self._input_std
        raw = _tanh_mlp_forward(normalized, self._mlp_weights, self._mlp_biases)
        squashed = np.tanh(raw)
        command = np.asarray(
            [
                0.5 * (squashed[0] + 1.0) * float(self.config.vx_limit_mps),
                squashed[1] * float(self.config.vy_limit_mps),
                squashed[2] * float(self.config.yaw_rate_limit_radps),
            ],
            dtype=np.float32,
        )
        return command
