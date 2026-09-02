# KU SPARCy — Emirates Robotics Competition 2026

Private development repository for Khalifa University's KU SPARCy team.

## Required evaluation command

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

## Implemented through Day 2

- Day 1 regression fixture: odometry-controlled clockwise 90-degree opening
  turn, upward head motion, live RGB verification, and simulation-clock timing.
- Day 2 competition entry node: continuous RGB shelf-marker perception during
  motion, official-texture shape classification, multi-frame confirmation,
  publication to `/erc/shelf_column_identification`, and a live timestamped
  annotated target-column image under `src/ku_sparcy_erc/erc_images/`.
- `phase1_fast_start:=true` preserves the validated Phase 1 opening maneuver.
- `phase1_fast_start:=false` performs an orientation-independent clockwise
  visual sweep and aligns the camera with the confirmed requested marker.

The solution reads only live robot topics and public model assets. It does not
read the randomized Gazebo entity names/order, change official competition
files, or modify the robot model, controllers, world, ROS distribution, or
Gazebo version.

See [`src/ku_sparcy_erc/README.md`](src/ku_sparcy_erc/README.md) for interfaces,
parameters, validation commands, and generated artifacts.
