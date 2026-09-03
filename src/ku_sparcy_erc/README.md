# `ku_sparcy_erc`

KU SPARCy's ROS 2 Humble solution package for ERC 2026.

## Competition entry point

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

The default performs:

```text
validated Phase-1 opening
  -> continuous Day-2 marker confirmation
  -> official Int32 publication + target-column image
  -> CameraInfo/depth/TF target geometry
  -> front-LiDAR-constrained holonomic approach
  -> controlled stand-off + quantitative image/JSON evidence
```

## Day 3 sensor interfaces

Subscriptions added by `day3_mission.py`:

- `/head_front_camera/head_front_camera/color/camera_info`
  (`sensor_msgs/msg/CameraInfo`)
- `/head_front_camera/head_front_camera/depth/image_rect_raw`
  (`sensor_msgs/msg/Image`)
- `/head_front_camera/head_front_camera/depth/camera_info`
  (`sensor_msgs/msg/CameraInfo`)
- `/scan_front_raw` (`sensor_msgs/msg/LaserScan`)
- TF from the depth optical and front-LiDAR frames to `base_footprint`

The point-cloud reconstruction is not required. The official RGB/depth streams
are matched and the raw depth image is sufficient for marker-centred range.

Publications remain:

- `/cmd_vel` (`geometry_msgs/msg/Twist`)
- `/head_controller/joint_trajectory`
- `/erc/shelf_column_identification` (`std_msgs/msg/Int32`)
- `/ku_sparcy/mission_status`
- `/ku_sparcy/marker_debug`
- `/ku_sparcy/approach_debug`

## Bearing and range

For target pixel `u`, the RGB optical bearing is:

```text
bearing_right = atan2(u - cx, fx)
```

The RGB bounding box is mapped into depth pixels through the two CameraInfo
models. Valid depth values are filtered to 0.2–8.0 m, the nearest coherent
surface is selected with a lower-quantile foreground band, and median/MAD
filtering rejects edge outliers. The selected optical point is:

```text
Xright = (u - cx) * Z / fx
Ydown  = (v - cy) * Z / fy
Zfront = Z
```

A live TF transform converts that point into `base_footprint`, where +x is
forward and +y is left. The planar base bearing is `atan2(y, x)`.

## LiDAR safety and controller

Every valid `/scan_front_raw` ray is transformed to `base_footprint`. Only
points ahead of the robot and within a 0.38 m half-width swept corridor are
retained. The 10th-percentile forward distance supplies robust clearance; a
three-point cluster inside 0.72 m triggers an immediate zero-velocity safety
hold.

The controller commands bounded `linear.x`, `linear.y`, and `angular.z`
simultaneously. A target to the left produces positive lateral strafe. Yaw
control holds the validated post-opening shelf-facing heading, so TIAGo reaches
an edge column without finishing at a diagonal shelf angle. Forward speed is
reduced by target-bearing magnitude and LiDAR clearance. Stale vision, depth, LiDAR, synchronization,
or TF never permits blind translation.

## Test modes

```bash
# Bearing/depth/LiDAR only after the normal opening and target detection
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=3 book_colour:=red \
  approach_motion_enabled:=false

# Slow, deliberately short approach
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=3 book_colour:=red \
  approach_distance_limit_m:=0.45 \
  approach_max_forward_speed_mps:=0.15

# Frozen Day 2 full Phase-1 regression
ros2 launch ku_sparcy_erc day2_regression.launch.py \
  shelf_column_number:=3 book_colour:=red
```

## Runtime artifacts

- `day3_result*.json`: atomic quantitative result.
- `erc_images/shelf_column_...png`: inherited live Day-2 identification image.
- `erc_images/shelf_approach_column_...png`: final live range/stand-off image.
- `day3_results/*.json`: ignored regression results.

All runtime images and JSON files are ignored by Git.

## Validation

```bash
python3 tools/test_day3_geometry.py
python3 tools/day3_interface_probe.py
./tools/day3_ros_healthcheck.sh
python3 tools/validate_day3_result.py day3_result.json \
  --target 3 --mode full --min-travel-m 0.75
./tools/run_day3_small_regression.sh
python3 tools/summarize_day3_results.py --expected-count 2
```

The Day 1 four-file fixture and the Day 2 detector, mission node, configuration,
and validators are frozen and checked against commit
`8c921d7ec228297efe683f7eb973ecb55973eaf0`.
