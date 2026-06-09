#!/usr/bin/env python3
"""Drive the low-level Go2 policy around the oval track with keyboard input.

This is a local interactive tool intended for Macs/desktops, not Colab.
It lets you:

- load a trained low-level checkpoint
- reset the robot onto the official oval track
- control the high-level command [vx, vy, yaw_rate] with the keyboard
- watch live visual feedback plus a top-down track map
- log (track_observation, manual_command) pairs for imitation learning
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from go2_pg_env.track import StandardOvalTrack
from run_track_bonus import _reset_lowlevel_on_track
from test_policy import load_policy_with_workaround
from track_bonus.controller_interface import build_track_controller_observation
from track_bonus.official_track import official_track


ROOT = Path(__file__).resolve().parent


@dataclass
class SessionBuffers:
    track_observation: list[np.ndarray]
    command: list[np.ndarray]
    qpos: list[np.ndarray]
    timestamp_sec: list[float]
    lap_fraction: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to a low-level PPO best_checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "manual_control", help="Where to save the session dataset.")
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--start-s-m", type=float, default=0.0, help="Track progress in meters to spawn at.")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU before restoring the checkpoint.")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument("--render-camera", type=str, default="track")
    parser.add_argument("--display-fps", type=int, default=12)
    parser.add_argument("--sim-steps-per-frame", type=int, default=2, help="How many control steps to simulate per displayed frame.")
    parser.add_argument("--max-vx-mps", type=float, default=1.2)
    parser.add_argument("--max-reverse-vx-mps", type=float, default=0.4)
    parser.add_argument("--max-vy-mps", type=float, default=0.35)
    parser.add_argument("--max-yaw-rate-radps", type=float, default=0.8)
    parser.add_argument("--vx-ramp-mps-per-sec", type=float, default=2.0)
    parser.add_argument("--vy-ramp-mps-per-sec", type=float, default=1.5)
    parser.add_argument("--yaw-ramp-radps-per-sec", type=float, default=3.0)
    return parser.parse_args()


def _make_env(stack: dict[str, Any], course_cfg: dict[str, Any], stage_name: str) -> Any:
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    env_name = course_cfg["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, course_cfg["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, course_cfg, stage_name)
    env_cfg.noise_config.level = 0.0
    env_cfg.pert_config.enable = False
    return registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(course_cfg))


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def _ramp(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def _keyboard_target(keys: Any, args: argparse.Namespace) -> np.ndarray:
    import pygame

    forward = float(keys[pygame.K_w] or keys[pygame.K_UP])
    reverse = float(keys[pygame.K_s] or keys[pygame.K_DOWN])
    left_yaw = float(keys[pygame.K_a] or keys[pygame.K_LEFT])
    right_yaw = float(keys[pygame.K_d] or keys[pygame.K_RIGHT])
    left_vy = float(keys[pygame.K_q])
    right_vy = float(keys[pygame.K_e])

    vx = forward * float(args.max_vx_mps) - reverse * float(args.max_reverse_vx_mps)
    vy = left_vy * float(args.max_vy_mps) - right_vy * float(args.max_vy_mps)
    yaw = left_yaw * float(args.max_yaw_rate_radps) - right_yaw * float(args.max_yaw_rate_radps)
    return np.asarray([vx, vy, yaw], dtype=np.float32)


def _to_surface(frame: np.ndarray) -> Any:
    import pygame

    rgb = np.ascontiguousarray(frame[:, :, :3])
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")


def _track_extents(track: StandardOvalTrack) -> tuple[float, float]:
    extent_x = track.straight_length_m / 2.0 + track.turn_radius_m + track.half_width_m + 2.0
    extent_y = track.turn_radius_m + track.half_width_m + 2.0
    return float(extent_x), float(extent_y)


def _world_to_map(point_xy: np.ndarray, *, track: StandardOvalTrack, map_size: int) -> tuple[int, int]:
    extent_x, extent_y = _track_extents(track)
    x = float(point_xy[0])
    y = float(point_xy[1])
    px = (x + extent_x) / (2.0 * extent_x)
    py = 1.0 - (y + extent_y) / (2.0 * extent_y)
    return int(np.clip(px, 0.0, 1.0) * (map_size - 1)), int(np.clip(py, 0.0, 1.0) * (map_size - 1))


def _draw_track_map(
    *,
    screen: Any,
    origin_x: int,
    origin_y: int,
    map_size: int,
    track: StandardOvalTrack,
    qpos: np.ndarray,
    command: np.ndarray,
    track_obs: np.ndarray,
    status_lines: list[str],
) -> None:
    import pygame

    panel = pygame.Surface((map_size, map_size + 140))
    panel.fill((18, 20, 24))
    track_surface = pygame.Surface((map_size, map_size))
    track_surface.fill((28, 31, 36))

    center_points = []
    outer_points = []
    inner_points = []
    for idx in range(180):
        s = track.length_m * idx / 180.0
        center, heading, _ = track.centerline_pose(s)
        normal = np.asarray([-np.sin(heading), np.cos(heading)], dtype=np.float64)
        center_points.append(_world_to_map(center, track=track, map_size=map_size))
        outer_points.append(_world_to_map(center + track.half_width_m * normal, track=track, map_size=map_size))
        inner_points.append(_world_to_map(center - track.half_width_m * normal, track=track, map_size=map_size))

    pygame.draw.lines(track_surface, (220, 220, 220), True, center_points, 2)
    pygame.draw.lines(track_surface, (65, 68, 74), True, outer_points, 4)
    pygame.draw.lines(track_surface, (65, 68, 74), True, inner_points, 4)

    robot_xy = np.asarray(qpos[:2], dtype=np.float32)
    robot_px, robot_py = _world_to_map(robot_xy, track=track, map_size=map_size)
    pygame.draw.circle(track_surface, (39, 146, 255), (robot_px, robot_py), 7)

    heading = float(np.arctan2(2.0 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]), 1.0 - 2.0 * (qpos[5] ** 2 + qpos[6] ** 2)))
    arrow_len = 18
    arrow_end = (
        int(robot_px + arrow_len * np.cos(heading)),
        int(robot_py - arrow_len * np.sin(heading)),
    )
    pygame.draw.line(track_surface, (255, 208, 0), (robot_px, robot_py), arrow_end, 3)

    panel.blit(track_surface, (0, 0))

    font = pygame.font.SysFont("Menlo", 18)
    small_font = pygame.font.SysFont("Menlo", 16)
    header = font.render("Manual Track Control", True, (240, 240, 240))
    panel.blit(header, (10, map_size + 8))
    details = [
        f"cmd vx={command[0]:+.2f} vy={command[1]:+.2f} yaw={command[2]:+.2f}",
        f"lap={track_obs[0]:.3f} lateral={track_obs[1]:+.3f} margin={track_obs[2]:+.3f}",
        f"heading_err={track_obs[3]:+.3f} curvature={track_obs[4]:+.3f}",
    ]
    for idx, line in enumerate(details):
        panel.blit(small_font.render(line, True, (220, 220, 220)), (10, map_size + 34 + idx * 20))
    for idx, line in enumerate(status_lines):
        panel.blit(small_font.render(line, True, (180, 220, 180)), (10, map_size + 94 + idx * 18))

    screen.blit(panel, (origin_x, origin_y))


def _save_session(output_dir: Path, buffers: SessionBuffers, session_summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "manual_dataset.npz",
        track_observation=np.asarray(buffers.track_observation, dtype=np.float32),
        command=np.asarray(buffers.command, dtype=np.float32),
        qpos=np.asarray(buffers.qpos, dtype=np.float32),
        timestamp_sec=np.asarray(buffers.timestamp_sec, dtype=np.float32),
        lap_fraction=np.asarray(buffers.lap_fraction, dtype=np.float32),
    )
    save_json(output_dir / "session_summary.json", session_summary)


def main() -> None:
    args = parse_args()

    if platform.system() == "Darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")

    course_cfg = load_json(args.config)
    course_cfg["runtime_overrides"] = {}
    if args.force_cpu:
        course_cfg["force_cpu"] = True
        course_cfg["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(course_cfg.get("force_cpu")) or bool(course_cfg["runtime_overrides"].get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("This tool requires pygame. Install it with `python -m pip install pygame`.") from exc

    stack = lazy_import_stack()
    jax = stack["jax"]
    env = _make_env(stack, course_cfg, args.stage_name)
    track = official_track()
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)
        step_fn = jax.jit(env.step)
    else:
        step_fn = env.step

    def reset_state(seed_offset: int = 0) -> Any:
        rng = jax.random.PRNGKey(int(args.seed) + seed_offset)
        return _reset_lowlevel_on_track(
            stack=stack,
            env=env,
            rng=rng,
            track=track,
            start_s=float(args.start_s_m),
        )

    pygame.init()
    pygame.font.init()
    map_size = max(360, int(args.render_height))
    screen = pygame.display.set_mode((int(args.render_width) + map_size, max(int(args.render_height), map_size + 140)))
    pygame.display.set_caption("Go2 Manual Track Control")
    clock = pygame.time.Clock()

    state = reset_state()
    command = np.zeros(3, dtype=np.float32)
    start_time = time.monotonic()
    reset_count = 0
    recording = True
    buffers = SessionBuffers(track_observation=[], command=[], qpos=[], timestamp_sec=[], lap_fraction=[])
    status_lines = [
        "W/S: forward/reverse  A/D: yaw left/right",
        "Q/E: lateral left/right  SPACE: zero  R: reset  TAB: toggle record  ESC: quit",
    ]

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        command[:] = 0.0
                    elif event.key == pygame.K_r:
                        reset_count += 1
                        state = reset_state(seed_offset=reset_count)
                        command[:] = 0.0
                    elif event.key == pygame.K_TAB:
                        recording = not recording

            keys = pygame.key.get_pressed()
            target = _keyboard_target(keys, args)
            dt = float(env.dt)
            command[0] = _ramp(command[0], float(target[0]), float(args.vx_ramp_mps_per_sec) * dt)
            command[1] = _ramp(command[1], float(target[1]), float(args.vy_ramp_mps_per_sec) * dt)
            command[2] = _ramp(command[2], float(target[2]), float(args.yaw_ramp_radps_per_sec) * dt)

            out_of_bounds = False
            terminated = False
            track_obs_array = None
            for _ in range(int(args.sim_steps_per_frame)):
                qpos_now = np.asarray(state.data.qpos, dtype=np.float32)
                track_obs = build_track_controller_observation(qpos=qpos_now, track=track)
                track_obs_array = track_obs.as_array()
                if recording:
                    buffers.track_observation.append(track_obs_array.copy())
                    buffers.command.append(command.copy())
                    buffers.qpos.append(qpos_now.copy())
                    buffers.timestamp_sec.append(time.monotonic() - start_time)
                    buffers.lap_fraction.append(float(track_obs.lap_fraction))
                state = _force_command(state, command, jax)
                state.info["rng"], act_key = jax.random.split(state.info["rng"])
                action, _ = policy(state.obs, act_key)
                state = step_fn(state, action)
                state = _force_command(state, command, jax)

                projection = track.project_xy_to_track(np.asarray(state.data.qpos[:2], dtype=np.float32))
                out_of_bounds = bool(projection.out_of_bounds)
                terminated = bool(np.asarray(state.done))
                if out_of_bounds or terminated:
                    break

            frame = env.render([state], height=int(args.render_height), width=int(args.render_width), camera=args.render_camera)[0]
            screen.fill((10, 12, 16))
            screen.blit(_to_surface(frame), (0, 0))
            if track_obs_array is None:
                qpos_now = np.asarray(state.data.qpos, dtype=np.float32)
                track_obs_array = build_track_controller_observation(qpos=qpos_now, track=track).as_array()
            status = status_lines + [f"recording={recording} resets={reset_count}"]
            if terminated:
                status.append("episode terminated: press R to reset")
            if out_of_bounds:
                status.append("out of bounds: press R to reset")
            _draw_track_map(
                screen=screen,
                origin_x=int(args.render_width),
                origin_y=0,
                map_size=map_size,
                track=track,
                qpos=np.asarray(state.data.qpos, dtype=np.float32),
                command=command,
                track_obs=track_obs_array,
                status_lines=status,
            )
            pygame.display.flip()
            clock.tick(int(args.display_fps))
    finally:
        summary = {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "config": str(args.config.resolve()),
            "stage_name": args.stage_name,
            "seed": int(args.seed),
            "num_samples": len(buffers.command),
            "recording_enabled_on_exit": recording,
            "resets": reset_count,
            "max_vx_mps": float(args.max_vx_mps),
            "max_vy_mps": float(args.max_vy_mps),
            "max_yaw_rate_radps": float(args.max_yaw_rate_radps),
        }
        _save_session(args.output_dir.resolve(), buffers, summary)
        pygame.quit()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
