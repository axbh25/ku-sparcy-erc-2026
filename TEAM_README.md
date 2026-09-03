# KU SPARCy — Emirates Robotics Competition 2026

Private development repository for Khalifa University's KU SPARCy team.

## Required evaluation command

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

## Implemented through Day 3

- **Day 1 frozen regression:** odometry-controlled clockwise 90-degree Phase-1
  opening, upward head motion, live RGB validation, and simulation-clock timing.
- **Day 2 frozen perception core:** deterministic live shelf-marker recognition,
  multi-frame confirmation, publication to `/erc/shelf_column_identification`,
  timestamped target-column evidence, and orientation-independent visual search.
- **Day 3 shelf approach:** actual CameraInfo intrinsics, aligned raw-depth
  sampling, TF projection into `base_footprint`, independent front-LiDAR swept
  corridor safety, and bounded simultaneous forward/lateral/yaw control to a
  deliberate stand-off distance.

The Day 3 node subclasses the committed Day 2 `MissionStart` class; it does not
replace or duplicate the validated marker detector. Motion, sensor freshness,
and recovery deadlines use Gazebo `/clock`. Wall time is recorded separately
and used only for startup/stall watchdogs.

No competition code reads `ERC_SEED`, randomized Gazebo entity names/order,
world poses, or expected-layout files. No official world, robot model,
controller, Docker file, Gazebo version, ROS distribution, arm, gripper, or torso
behavior is modified.

See [`src/ku_sparcy_erc/README.md`](src/ku_sparcy_erc/README.md) for interfaces,
parameters, validation modes, and runtime artifacts.
