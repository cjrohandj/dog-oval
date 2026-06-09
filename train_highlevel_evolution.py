#!/usr/bin/env python3
"""Optimize a learned MLP planner with a simple evolutionary method.

This script searches directly over the learned planner weights using rollout
score from `run_track_bonus.py`. It uses a lightweight CEM-style update:

1. keep a Gaussian over flattened MLP weights
2. sample a population of candidates
3. evaluate each candidate with the track bonus rollout
4. update the Gaussian from the top-scoring elites

The resulting artifacts match the `.npz`/`planner_config.json` format consumed
by `track_bonus.planner.StarterTrackPlanner` when `planner_type=learned_mlp`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from track_bonus.official_track import official_track_config


ROOT = Path(__file__).resolve().parent


@dataclass
class WeightsSpec:
    hidden_sizes: list[int]
    layer_shapes: list[tuple[int, int]]
    bias_shapes: list[tuple[int, ...]]
    input_mean: np.ndarray
    input_std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--base-planner-config", type=Path, default=ROOT / "configs" / "learned_planner.example.json")
    parser.add_argument("--init-weights", type=Path, default=None, help="Optional .npz produced by train_highlevel_mlp.py.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--parallel-evals", type=int, default=1)
    parser.add_argument("--eval-seconds", type=float, default=45.0)
    parser.add_argument("--init-sigma", type=float, default=0.08)
    parser.add_argument("--min-sigma", type=float, default=0.02)
    parser.add_argument("--sigma-decay", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--stand-seconds", type=float, default=0.0)
    parser.add_argument("--vx-limit-mps", type=float, default=1.2)
    parser.add_argument("--vy-limit-mps", type=float, default=0.4)
    parser.add_argument("--yaw-rate-limit-radps", type=float, default=0.7)
    return parser.parse_args()


def _layer_shapes(hidden_sizes: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, ...]]]:
    sizes = [5, *hidden_sizes, 3]
    weights = [(sizes[idx], sizes[idx + 1]) for idx in range(len(sizes) - 1)]
    biases = [(sizes[idx + 1],) for idx in range(len(sizes) - 1)]
    return weights, biases


def _default_init(spec: WeightsSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pieces: list[np.ndarray] = []
    for in_dim, out_dim in spec.layer_shapes:
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        pieces.append(rng.uniform(-limit, limit, size=(in_dim * out_dim,)).astype(np.float32))
    for bias_shape in spec.bias_shapes:
        pieces.append(np.zeros(int(np.prod(bias_shape)), dtype=np.float32))
    vector = np.concatenate(pieces, axis=0).astype(np.float32)
    return vector, np.full_like(vector, fill_value=1.0, dtype=np.float32)


def _load_init_weights(path: Path, spec: WeightsSpec) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    pieces: list[np.ndarray] = []
    for idx, shape in enumerate(spec.layer_shapes):
        key = f"W{idx}"
        pieces.append(np.asarray(payload[key], dtype=np.float32).reshape(shape).ravel())
    for idx, shape in enumerate(spec.bias_shapes):
        key = f"b{idx}"
        pieces.append(np.asarray(payload[key], dtype=np.float32).reshape(shape).ravel())
    input_mean = np.asarray(payload["input_mean"], dtype=np.float32) if "input_mean" in payload else np.zeros(5, dtype=np.float32)
    input_std = np.asarray(payload["input_std"], dtype=np.float32) if "input_std" in payload else np.ones(5, dtype=np.float32)
    return np.concatenate(pieces, axis=0).astype(np.float32), input_mean, np.maximum(input_std, 1e-6)


def _unflatten(vector: np.ndarray, spec: WeightsSpec) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "input_mean": spec.input_mean.astype(np.float32),
        "input_std": spec.input_std.astype(np.float32),
    }
    cursor = 0
    for idx, shape in enumerate(spec.layer_shapes):
        size = int(np.prod(shape))
        payload[f"W{idx}"] = vector[cursor : cursor + size].reshape(shape).astype(np.float32)
        cursor += size
    for idx, shape in enumerate(spec.bias_shapes):
        size = int(np.prod(shape))
        payload[f"b{idx}"] = vector[cursor : cursor + size].reshape(shape).astype(np.float32)
        cursor += size
    if cursor != len(vector):
        raise ValueError(f"Unexpected vector length {len(vector)} for cursor {cursor}")
    return payload


def _write_candidate(
    *,
    candidate_dir: Path,
    vector: np.ndarray,
    spec: WeightsSpec,
    stand_seconds: float,
    vx_limit: float,
    vy_limit: float,
    yaw_limit: float,
) -> tuple[Path, Path]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    weights_path = candidate_dir / "planner_weights.npz"
    np.savez(weights_path, **_unflatten(vector, spec))
    planner_config = {
        "planner_type": "learned_mlp",
        "weights_path": str(weights_path.name),
        "stand_seconds": float(stand_seconds),
        "vx_limit_mps": float(vx_limit),
        "vy_limit_mps": float(vy_limit),
        "yaw_rate_limit_radps": float(yaw_limit),
        **official_track_config(),
    }
    planner_path = candidate_dir / "planner_config.json"
    planner_path.write_text(json.dumps(planner_config, indent=2), encoding="utf-8")
    return weights_path, planner_path


def _run_eval(
    *,
    checkpoint_dir: Path,
    planner_path: Path,
    config_path: Path,
    output_dir: Path,
    eval_seconds: float,
    force_cpu: bool,
) -> dict[str, float]:
    cmd = [
        sys.executable,
        "run_track_bonus.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--planner-config",
        str(planner_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--duration-seconds",
        str(eval_seconds),
        "--no-render",
    ]
    if force_cpu:
        cmd.append("--force-cpu")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        scores = payload["scores"]
        fitness = (
            0.60 * float(scores["completion_score"])
            + 0.20 * float(scores["speed_score"])
            + 0.15 * float(scores["line_keeping_score"])
            + 0.05 * float(scores["stability_score"])
        )
        metrics = payload["metrics"]
        if bool(metrics.get("fall")) or bool(metrics.get("boundary_violation")):
            fitness *= 0.55
        return {
            "fitness": float(fitness),
            "official_composite_score": float(scores["composite_score"]),
            "completion_score": float(scores["completion_score"]),
            "speed_score": float(scores["speed_score"]),
            "line_keeping_score": float(scores["line_keeping_score"]),
            "stability_score": float(scores["stability_score"]),
            "fall": float(bool(metrics.get("fall"))),
            "boundary_violation": float(bool(metrics.get("boundary_violation"))),
        }
    except Exception:
        return {
            "fitness": -1.0,
            "official_composite_score": -1.0,
            "completion_score": -1.0,
            "speed_score": -1.0,
            "line_keeping_score": -1.0,
            "stability_score": -1.0,
            "fall": -1.0,
            "boundary_violation": -1.0,
        }


def _evaluate_candidate(
    *,
    cand_idx: int,
    vector: np.ndarray,
    iteration: int,
    output_dir: Path,
    spec: WeightsSpec,
    checkpoint_dir: Path,
    config_path: Path,
    eval_seconds: float,
    force_cpu: bool,
    stand_seconds: float,
    vx_limit: float,
    vy_limit: float,
    yaw_limit: float,
) -> dict[str, Any]:
    candidate_dir = output_dir / "candidates" / f"iter_{iteration:02d}_cand_{cand_idx:02d}"
    _, planner_path = _write_candidate(
        candidate_dir=candidate_dir,
        vector=vector,
        spec=spec,
        stand_seconds=stand_seconds,
        vx_limit=vx_limit,
        vy_limit=vy_limit,
        yaw_limit=yaw_limit,
    )
    eval_result = _run_eval(
        checkpoint_dir=checkpoint_dir,
        planner_path=planner_path,
        config_path=config_path,
        output_dir=candidate_dir / "eval",
        eval_seconds=eval_seconds,
        force_cpu=force_cpu,
    )
    return {
        "candidate": cand_idx,
        "fitness": float(eval_result["fitness"]),
        "official_composite_score": float(eval_result["official_composite_score"]),
        "completion_score": float(eval_result["completion_score"]),
        "speed_score": float(eval_result["speed_score"]),
        "line_keeping_score": float(eval_result["line_keeping_score"]),
        "stability_score": float(eval_result["stability_score"]),
        "fall": float(eval_result["fall"]),
        "boundary_violation": float(eval_result["boundary_violation"]),
        "candidate_dir": str(candidate_dir),
        "planner_path": str(planner_path),
        "vector": vector,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_shapes, bias_shapes = _layer_shapes(list(args.hidden_sizes))
    dummy_spec = WeightsSpec(
        hidden_sizes=list(args.hidden_sizes),
        layer_shapes=layer_shapes,
        bias_shapes=bias_shapes,
        input_mean=np.zeros(5, dtype=np.float32),
        input_std=np.ones(5, dtype=np.float32),
    )

    if args.init_weights is not None:
        mean_vector, input_mean, input_std = _load_init_weights(args.init_weights.resolve(), dummy_spec)
    else:
        mean_vector, _ = _default_init(dummy_spec, rng)
        input_mean = np.zeros(5, dtype=np.float32)
        input_std = np.ones(5, dtype=np.float32)

    spec = WeightsSpec(
        hidden_sizes=list(args.hidden_sizes),
        layer_shapes=layer_shapes,
        bias_shapes=bias_shapes,
        input_mean=input_mean,
        input_std=input_std,
    )
    sigma = np.full_like(mean_vector, fill_value=float(args.init_sigma), dtype=np.float32)

    best_score = -1.0
    best_vector = mean_vector.copy()
    history: list[dict[str, Any]] = []

    for iteration in range(int(args.iterations)):
        candidates: list[np.ndarray] = [mean_vector.copy()]
        while len(candidates) < int(args.population):
            noise = rng.normal(0.0, 1.0, size=mean_vector.shape).astype(np.float32)
            candidates.append((mean_vector + sigma * noise).astype(np.float32))

        records: list[dict[str, Any]] = []
        parallel_evals = max(1, int(args.parallel_evals))
        eval_kwargs = {
            "iteration": iteration,
            "output_dir": output_dir,
            "spec": spec,
            "checkpoint_dir": args.checkpoint_dir.resolve(),
            "config_path": args.config.resolve(),
            "eval_seconds": float(args.eval_seconds),
            "force_cpu": bool(args.force_cpu),
            "stand_seconds": float(args.stand_seconds),
            "vx_limit": float(args.vx_limit_mps),
            "vy_limit": float(args.vy_limit_mps),
            "yaw_limit": float(args.yaw_rate_limit_radps),
        }
        if parallel_evals == 1:
            eval_results = [
                _evaluate_candidate(cand_idx=cand_idx, vector=vector, **eval_kwargs)
                for cand_idx, vector in enumerate(candidates)
            ]
        else:
            eval_results = []
            with ThreadPoolExecutor(max_workers=parallel_evals) as executor:
                futures = [
                    executor.submit(_evaluate_candidate, cand_idx=cand_idx, vector=vector, **eval_kwargs)
                    for cand_idx, vector in enumerate(candidates)
                ]
                for future in as_completed(futures):
                    eval_results.append(future.result())

        eval_results.sort(key=lambda item: int(item["candidate"]))
        for result in eval_results:
            cand_idx = int(result["candidate"])
            score = float(result["fitness"])
            vector = np.asarray(result["vector"], dtype=np.float32)
            record = {
                "candidate": cand_idx,
                "fitness": score,
                "official_composite_score": float(result["official_composite_score"]),
                "completion_score": float(result["completion_score"]),
                "speed_score": float(result["speed_score"]),
                "line_keeping_score": float(result["line_keeping_score"]),
                "stability_score": float(result["stability_score"]),
                "fall": float(result["fall"]),
                "boundary_violation": float(result["boundary_violation"]),
                "candidate_dir": result["candidate_dir"],
            }
            records.append(record)
            if score > best_score:
                best_score = score
                best_vector = vector.copy()
                best_dir = output_dir / "best"
                _, best_planner_path = _write_candidate(
                    candidate_dir=best_dir,
                    vector=best_vector,
                    spec=spec,
                    stand_seconds=float(args.stand_seconds),
                    vx_limit=float(args.vx_limit_mps),
                    vy_limit=float(args.vy_limit_mps),
                    yaw_limit=float(args.yaw_rate_limit_radps),
                )
                (output_dir / "best_score.json").write_text(
                    json.dumps(
                        {
                            "fitness": best_score,
                            "official_composite_score": float(result["official_composite_score"]),
                            "planner_config": str(best_planner_path),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            print(
                f"iter={iteration} cand={cand_idx} fitness={score:.4f} "
                f"official={float(result['official_composite_score']):.4f} best={best_score:.4f}",
                flush=True,
            )

        ranked = sorted(zip(candidates, records), key=lambda item: item[1]["fitness"], reverse=True)
        elite = ranked[: max(1, min(int(args.elite_count), len(ranked)))]
        elite_vectors = np.asarray([item[0] for item in elite], dtype=np.float32)
        mean_vector = elite_vectors.mean(axis=0).astype(np.float32)
        elite_std = elite_vectors.std(axis=0).astype(np.float32)
        sigma = np.maximum(
            float(args.min_sigma),
            float(args.sigma_decay) * sigma + (1.0 - float(args.sigma_decay)) * elite_std,
        ).astype(np.float32)
        iteration_summary = {
            "iteration": iteration,
            "best_fitness": float(best_score),
            "mean_fitness": float(np.mean([record["fitness"] for record in records])),
            "elite_mean_fitness": float(np.mean([item[1]["fitness"] for item in elite])),
            "mean_official_composite_score": float(np.mean([record["official_composite_score"] for record in records])),
            "sigma_mean": float(np.mean(sigma)),
            "records": records,
        }
        history.append(iteration_summary)
        (output_dir / "search_summary.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "config_path": str(args.config.resolve()),
        "init_weights": None if args.init_weights is None else str(args.init_weights.resolve()),
        "iterations": int(args.iterations),
        "population": int(args.population),
        "elite_count": int(args.elite_count),
        "eval_seconds": float(args.eval_seconds),
        "fitness_formula": "0.60*completion_score + 0.20*speed_score + 0.15*line_keeping_score + 0.05*stability_score",
        "best_fitness": float(best_score),
        "best_planner_config": str(output_dir / "best" / "planner_config.json"),
        "best_weights": str(output_dir / "best" / "planner_weights.npz"),
    }
    (output_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
