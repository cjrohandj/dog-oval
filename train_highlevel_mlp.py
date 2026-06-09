#!/usr/bin/env python3
"""Train a small learned high-level planner for the oval track.

This trainer is intentionally lightweight so it runs in a plain Colab/runtime
without extra ML dependencies. It distills the hand-written starter planner
into a tiny tanh MLP:

    track_observation[5] -> MLP -> [vx, vy, yaw_rate]

The produced weights are saved in the `.npz` format expected by
`track_bonus.planner.StarterTrackPlanner` when `planner_type="learned_mlp"`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from go2_pg_env.track import StandardOvalTrack
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track_config
from track_bonus.planner import StarterPlannerConfig, StarterTrackPlanner


ROOT = Path(__file__).resolve().parent


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-planner-config", type=Path, default=ROOT / "configs" / "starter_planner.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--train-samples", type=int, default=40000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stand-seconds", type=float, default=0.0)
    parser.add_argument("--vx-limit-mps", type=float, default=1.2)
    parser.add_argument("--vy-limit-mps", type=float, default=0.4)
    parser.add_argument("--yaw-rate-limit-radps", type=float, default=0.7)
    return parser.parse_args()


def _sample_observations(track: StandardOvalTrack, count: int, rng: np.random.Generator) -> np.ndarray:
    obs = np.zeros((count, 5), dtype=np.float32)
    for idx in range(count):
        s = float(rng.uniform(0.0, track.length_m))
        _, _, curvature = track.centerline_pose(s)
        lateral_error_norm = float(rng.uniform(-0.85, 0.85))
        boundary_margin_norm = float(np.clip(1.0 - abs(lateral_error_norm), -0.15, 1.0))
        heading_error_rad = float(rng.uniform(-0.9, 0.9))
        obs[idx] = np.asarray(
            [
                (s % track.length_m) / track.length_m,
                lateral_error_norm,
                boundary_margin_norm,
                heading_error_rad,
                curvature * track.turn_radius_m,
            ],
            dtype=np.float32,
        )
    return obs


def _planner_targets(planner: StarterTrackPlanner, observations: np.ndarray) -> np.ndarray:
    labels = np.zeros((observations.shape[0], 3), dtype=np.float32)
    for idx, row in enumerate(observations):
        obs = TrackControllerObservation(
            lap_fraction=float(row[0]),
            lateral_error_norm=float(row[1]),
            boundary_margin_norm=float(row[2]),
            heading_error_rad=float(row[3]),
            curvature_norm=float(row[4]),
        )
        labels[idx] = planner.command_from_observation(obs)
    return labels


def build_dataset(
    *,
    planner: StarterTrackPlanner,
    track: StandardOvalTrack,
    train_samples: int,
    val_samples: int,
    rng: np.random.Generator,
) -> DatasetBundle:
    x_train = _sample_observations(track, train_samples, rng)
    y_train = _planner_targets(planner, x_train)
    x_val = _sample_observations(track, val_samples, rng)
    y_val = _planner_targets(planner, x_val)
    return DatasetBundle(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)


def init_mlp(layer_sizes: Iterable[int], rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sizes = list(layer_sizes)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weights.append(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32))
        biases.append(np.zeros(out_dim, dtype=np.float32))
    return weights, biases


def mlp_forward(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    activations = [x]
    preacts: list[np.ndarray] = []
    hidden = x
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        preact = hidden @ weight + bias
        preacts.append(preact)
        if idx + 1 == len(weights):
            hidden = preact
        else:
            hidden = np.tanh(preact)
        activations.append(hidden)
    return hidden, activations, preacts


def bounded_outputs(
    raw: np.ndarray,
    *,
    vx_limit: float,
    vy_limit: float,
    yaw_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    squashed = np.tanh(raw)
    outputs = np.empty_like(squashed)
    outputs[:, 0] = 0.5 * (squashed[:, 0] + 1.0) * vx_limit
    outputs[:, 1] = squashed[:, 1] * vy_limit
    outputs[:, 2] = squashed[:, 2] * yaw_limit
    return outputs, squashed


def train_epoch(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    learning_rate: float,
    vx_limit: float,
    vy_limit: float,
    yaw_limit: float,
) -> float:
    raw, activations, _ = mlp_forward(x_batch, weights, biases)
    pred, squashed = bounded_outputs(raw, vx_limit=vx_limit, vy_limit=vy_limit, yaw_limit=yaw_limit)
    diff = pred - y_batch
    loss = float(np.mean(np.square(diff)))

    grad_pred = (2.0 / diff.size) * diff
    grad_raw = np.empty_like(grad_pred)
    tanh_prime = 1.0 - np.square(squashed)
    grad_raw[:, 0] = grad_pred[:, 0] * (0.5 * vx_limit) * tanh_prime[:, 0]
    grad_raw[:, 1] = grad_pred[:, 1] * vy_limit * tanh_prime[:, 1]
    grad_raw[:, 2] = grad_pred[:, 2] * yaw_limit * tanh_prime[:, 2]

    grad_hidden = grad_raw
    for layer_idx in reversed(range(len(weights))):
        layer_input = activations[layer_idx]
        grad_w = layer_input.T @ grad_hidden
        grad_b = np.sum(grad_hidden, axis=0)
        if layer_idx > 0:
            backprop = grad_hidden @ weights[layer_idx].T
            grad_hidden = backprop * (1.0 - np.square(activations[layer_idx]))
        weights[layer_idx] = weights[layer_idx] - learning_rate * grad_w.astype(np.float32)
        biases[layer_idx] = biases[layer_idx] - learning_rate * grad_b.astype(np.float32)
    return loss


def eval_loss(
    x: np.ndarray,
    y: np.ndarray,
    *,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    vx_limit: float,
    vy_limit: float,
    yaw_limit: float,
) -> float:
    raw, _, _ = mlp_forward(x, weights, biases)
    pred, _ = bounded_outputs(raw, vx_limit=vx_limit, vy_limit=vy_limit, yaw_limit=yaw_limit)
    return float(np.mean(np.square(pred - y)))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_config = StarterPlannerConfig.load(args.base_planner_config)
    teacher = StarterTrackPlanner(teacher_config)
    track = teacher.track

    dataset = build_dataset(
        planner=teacher,
        track=track,
        train_samples=int(args.train_samples),
        val_samples=int(args.val_samples),
        rng=rng,
    )
    input_mean = dataset.x_train.mean(axis=0).astype(np.float32)
    input_std = np.maximum(dataset.x_train.std(axis=0), 1e-6).astype(np.float32)
    x_train = ((dataset.x_train - input_mean) / input_std).astype(np.float32)
    x_val = ((dataset.x_val - input_mean) / input_std).astype(np.float32)
    y_train = dataset.y_train.astype(np.float32)
    y_val = dataset.y_val.astype(np.float32)

    layer_sizes = [5, *list(args.hidden_sizes), 3]
    weights, biases = init_mlp(layer_sizes, rng)

    history: list[dict[str, float]] = []
    best = {
        "loss": float("inf"),
        "weights": [weight.copy() for weight in weights],
        "biases": [bias.copy() for bias in biases],
    }

    for step in range(1, int(args.steps) + 1):
        batch_idx = rng.integers(0, x_train.shape[0], size=int(args.batch_size))
        train_loss = train_epoch(
            x_train[batch_idx],
            y_train[batch_idx],
            weights=weights,
            biases=biases,
            learning_rate=float(args.learning_rate),
            vx_limit=float(args.vx_limit_mps),
            vy_limit=float(args.vy_limit_mps),
            yaw_limit=float(args.yaw_rate_limit_radps),
        )
        if step == 1 or step % 100 == 0 or step == int(args.steps):
            val_loss = eval_loss(
                x_val,
                y_val,
                weights=weights,
                biases=biases,
                vx_limit=float(args.vx_limit_mps),
                vy_limit=float(args.vy_limit_mps),
                yaw_limit=float(args.yaw_rate_limit_radps),
            )
            history.append({"step": float(step), "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best["loss"]:
                best = {
                    "loss": val_loss,
                    "weights": [weight.copy() for weight in weights],
                    "biases": [bias.copy() for bias in biases],
                }
            print(f"step={step} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

    weights_path = output_dir / "planner_weights.npz"
    payload: dict[str, np.ndarray] = {
        "input_mean": input_mean,
        "input_std": input_std,
    }
    for idx, weight in enumerate(best["weights"]):
        payload[f"W{idx}"] = weight
    for idx, bias in enumerate(best["biases"]):
        payload[f"b{idx}"] = bias
    np.savez(weights_path, **payload)

    planner_config = {
        "planner_type": "learned_mlp",
        "weights_path": str(weights_path.name),
        "stand_seconds": float(args.stand_seconds),
        "vx_limit_mps": float(args.vx_limit_mps),
        "vy_limit_mps": float(args.vy_limit_mps),
        "yaw_rate_limit_radps": float(args.yaw_rate_limit_radps),
        **official_track_config(),
    }
    config_path = output_dir / "planner_config.json"
    config_path.write_text(json.dumps(planner_config, indent=2), encoding="utf-8")

    summary = {
        "teacher_planner_config": str(args.base_planner_config.resolve()),
        "weights_path": str(weights_path),
        "planner_config_path": str(config_path),
        "hidden_sizes": list(args.hidden_sizes),
        "train_samples": int(args.train_samples),
        "val_samples": int(args.val_samples),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "best_val_loss": float(best["loss"]),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
