# `ku_sparcy_erc`

KU SPARCy's ROS 2 Humble solution package for the Emirates Robotics Competition
2026 library-assistant task.

## Entry point

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

Required arguments:

- `shelf_column_number`: requested marker label, integer 1–5.
- `book_colour`: `red`, `green`, `yellow`, or `blue`.

Useful Day 2 arguments:

- `phase1_fast_start:=true`: validated closed-loop clockwise 90-degree opening
  maneuver while perception runs continuously.
- `phase1_fast_start:=false`: orientation-independent visual search and marker
  alignment for later physical testing.
- `result_path:=...`: atomic JSON validation output.
- `image_output_dir:=...`: live annotated-image directory.
- `validation_require_all_markers:=true`: test-only; confirms 1–5 and waits for
  the complete left-to-right order. The competition default is `false`, so the
  robot does not wait after the requested marker is reliable.

## Day 2 architecture

`mission_start.py` owns the opening/search state machine and ROS interfaces.
`marker_detector.py` is ROS-independent and implements the deterministic CPU
pipeline:

1. Inspect the upper RGB region for dark connected components.
2. Require a bright marker-plate neighbourhood.
3. Normalize each candidate glyph.
4. Compare HOG, normalized-pixel, and contour-shape features against augmented
   official 1–5 marker textures installed by `erc_description`.
5. Suppress overlapping duplicates.
6. Confirm a digit only after spatially consistent observations in at least
   three distinct frames within the configured simulation-time window.

The randomized marker order is never read from Gazebo state. The official
textures are used only as visual templates; classification is applied to live
camera pixels.

## ROS interfaces

Subscriptions:

- `/clock` (`rosgraph_msgs/msg/Clock`)
- `/odom` (`nav_msgs/msg/Odometry`)
- `/joint_states` (`sensor_msgs/msg/JointState`)
- `/head_front_camera/head_front_camera/color/image_raw`
  (`sensor_msgs/msg/Image`)

Publications:

- `/cmd_vel` (`geometry_msgs/msg/Twist`)
- `/head_controller/joint_trajectory`
  (`trajectory_msgs/msg/JointTrajectory`)
- `/erc/shelf_column_identification` (`std_msgs/msg/Int32`)
- `/ku_sparcy/mission_status` (`std_msgs/msg/String`)
- `/ku_sparcy/marker_debug` (`std_msgs/msg/String`)


## Column-result interpretation

The current Phase 1 document specifies an `Int32` containing the shelf column
number but does not separately define a message for the physical left-to-right
index of a randomized marker. The competition node therefore publishes the
visually confirmed requested marker label (1–5), matching the literal example.
For navigation and organizer clarification, the result JSON also records the
bounding box and, when all five markers are visible in validation mode,
`target_column_index_left_to_right`.

## Runtime evidence

Default outputs:

- `day2_result.json`
- `erc_images/shelf_column_<digit>_sim_<time>_utc_<time>.png`
- optional matrix outputs under `day2_results/`

The annotated target-column image contains the target bounding box, target number,
confidence, confirming-frame count, simulation timestamp, UTC wall timestamp,
robot yaw, and state. Generated results and images are ignored by Git.

## Validation

```bash
# Official asset/dependency smoke test
python3 tools/test_marker_detector_assets.py

# One result
python3 tools/validate_day2_result.py day2_result.json \
  --target 2 --mode fast

# Full nine-case headless matrix
./tools/run_day2_matrix.sh
./tools/summarize_day2_results.py
```

`opening_sequence.py` and `config/opening_sequence.yaml` remain the frozen Day 1
regression fixture. Motion timeouts continue to use Gazebo simulation time;
wall time is recorded separately.
