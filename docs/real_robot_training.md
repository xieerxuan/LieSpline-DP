# Real-robot training

Record synchronized demonstrations containing **camera images, measured robot
states and commanded targets**. Measured states are model inputs; targets
provide action labels. Align samples to the task frequency (default 15 Hz)
before conversion; the reader does not resample recordings.

Store one demonstration per directory:

```text
data/real_robot/episodes/
└── episode_000/
    ├── steps.npz
    ├── meta.json
    └── images/          # Image files referenced by steps.npz
```

Each array in `steps.npz` has the same leading length `N`:

| Field | Shape and content |
| --- | --- |
| `timestamp`, `record_frame_index` | `[N]`: strictly increasing timestamps in seconds (float64) and frame indices (integer) |
| `active` | `[N]` boolean; the longest consecutive active segment is used (default minimum: 10 steps) |
| `tcp_actual` | `[N,6]`: measured absolute `[x,y,z,rx,ry,rz]`, position in meters and rotation vector in radians |
| `tcp_target_position`, `tcp_target_quat` | `[N,3]` absolute target position in meters and `[N,4]` target unit quaternion |
| `gripper_actual_q`, `gripper_target_q` (pose-and-gripper only) | `[N]` or `[N,1]` each: measured position for observations and intended target for action labels, in consistent gripper units |
| `wrist_rgb_path`, `global_rgb_path` | `[N]` Unicode strings: image paths relative to the episode directory, e.g. `images/wrist_000.png` |

For the target quaternions, set `meta.json` to
`{"tcp_target_quat_order": "xyzw"}` (or `"wxyz"`). Use a consistent reference
frame and TCP definition, finite numeric values, and configure gripper bounds
to match the recorded units. PNG/JPEG images are decoded as RGB and resized to
the configured size (default 240 × 320). All listed fields are required for
pose-and-gripper training; pose-only training requires no gripper fields.
Extra recording fields are ignored unless explicitly supported by the reader.

For pose-and-gripper LieSpline-DP training, conversion maps `gripper_target_q`
to binary open/close intent with hysteresis. It does not preserve arbitrary
intermediate gripper targets.

The reader also accepts `tcp_target` `[N,6]` in the same pose format as
`tcp_actual`, in place of the two target arrays, subject to its
[required metadata](../liespline_dp/dataset/real_robot_episode_reader.py).
Do not mix target representations in one episode. Optional tactile fields
are described in the same reader; include them consistently across episodes.

From the repository root, build the image cache, convert targets to spline
control points, then train:

```bash
python -m liespline_dp.scripts.build_real_robot_cache \
  --dataset-root data/real_robot/episodes
python -m liespline_dp.scripts.real_robot_bspline_se3_dataset_conversion
python train_real_robot.py \
  training.device=cuda:0 training.resume=false logging.mode=offline \
  hydra.run.dir=data/outputs/real_robot_liespline
```

These commands use the supplied pose-and-gripper task and cache paths under
`data/real_robot/cache/`. For pose-only training, select the matching
`baseline_image_pose_only.yaml` and `bspline_se3_image_pose_only.yaml` task
configs with `--task-config` for the two conversion commands, then pass
`task=real_robot/bspline_se3_image_pose_only` to training.
