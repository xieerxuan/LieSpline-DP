# Real-robot deployment

The repository provides model inference, action decoding and a reference execution client. Call the policy from your own controller or reuse the optional Robot Service client.

## Model interface

Load the checkpoint configuration, weights and normalizer, select the EMA
model when configured, and use `eval()` with `torch.inference_mode()`.
Call `policy.predict_action({"obs": obs})` to obtain an action chunk.
LieSpline-DP retains plan state between predictions. Call `policy.reset()`
before a new episode and after inference warmup; retain state between
consecutively executed chunks.

Shapes follow the checkpoint: `B` is batch size, `O` observation history length,
`N` action count, and `H/W` image dimensions.

| Field | Shape / meaning |
| --- | --- |
| Input camera fields (`wrist_rgb`, `global_rgb`) | `[B,O,3,H,W]`, RGB in `[0,1]` |
| Input `eef_pos`, `eef_quat` | `[B,O,3]` position in meters; `[B,O,4]` unit quaternion in `xyzw` order |
| Input `gripper_q`, when configured | `[B,O,1]`, measured gripper coordinate in training units |
| Output `action` | `[B,N,10]`: absolute position, rotation6d and gripper target; `[B,N,9]` for pose-only |
| Output `waypoint_pose` | `[B,N,4,4]`, absolute TCP transforms |
| Output `waypoint_grip`, when configured | `[B,N,1]`, normalized targets in `[-1,1]` |

Supply float32 tensors with synchronized history ordered oldest to newest.
Match the training coordinate frame, TCP definition and gripper mapping;
the policy applies normalization internally. `action` uses physical gripper
coordinates, while `waypoint_grip` is normalized. Rotation6d concatenates the
first two rows of the rotation matrix. `action_pred` contains control-point
parameters, not executable actions.

## Continuous execution

1. Predict a complete action chunk and start its execution.
2. At the configured inference trigger, collect new observations and predict the next chunk while the current chunk finishes.
3. Append the complete new chunk at the previous plan boundary. Its targets
   occur at offsets `1/f, 2/f, ..., N/f`, where `f` is the configured frequency.

The supplied defaults use 8 actions at 15 Hz and observations ending at step 6 for the next prediction.


## Connecting your hardware

Your application supplies images and robot state to the model, then sends the
predicted targets to your controller at the configured times. Choose either path:

- **Use your own control program.** Call `predict_action()` directly and handle
  observation collection and action execution yourself. You choose how the
  model and hardware communicate.
- **Reuse the provided execution script.** Supply a separate program that reads
  sensors and executes actions through the existing [service interface](../liespline_dp/real_world/robot_service_hardware_backend.py).
  This hardware-side program is not included in the repository.

For the second path, run [the evaluation script](../eval_real_robot_bspline_se3.py)
with `--mode shadow` to inspect predictions without sending actions, then
`--mode live` to execute them. Specify the model with `--input <checkpoint>`
and the service connection with `--backend-config <config.yaml>`; see the
[connection configuration](../liespline_dp/config/real_world/robot_service_backend.yaml).
