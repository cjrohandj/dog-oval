# Manual Track Control

Use [manual_track_control.py](/Users/school/Desktop/coding/dog-oval/manual_track_control.py) to drive the low-level checkpoint around the official oval with keyboard input on a local desktop or Mac.

## Install

```bash
python -m pip install pygame
```

If you are on macOS, the script defaults `MUJOCO_GL=glfw` to make local interactive rendering more likely to work.

## Run

```bash
python manual_track_control.py \
  --checkpoint-dir /path/to/best_checkpoint \
  --config configs/course_config.json \
  --stage-name stage_2 \
  --output-dir artifacts/manual_control
```

## Controls

- `W` / `Up`: increase forward `vx`
- `S` / `Down`: reverse / reduce `vx`
- `A` / `Left`: yaw left
- `D` / `Right`: yaw right
- `Q`: lateral left `vy`
- `E`: lateral right `vy`
- `Space`: zero the command
- `R`: reset back to the start of the oval
- `Tab`: toggle dataset recording
- `Esc`: quit and save

## Output

The script saves:

- `manual_dataset.npz`
- `session_summary.json`

inside the chosen output directory.

The dataset contains:

- `track_observation`
- `command`
- `qpos`
- `timestamp_sec`
- `lap_fraction`

You can later use those pairs as imitation-learning data for a learned planner.
