# KU SPARCy — Emirates Robotics Competition 2026

Independent KU SPARCy development repository.

## Required evaluation command

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

## Implemented through revised Day 4

- Frozen Day 1 closed-loop Phase-1 opening.
- Frozen Day 2 continuous shelf-marker perception and official column result.
- Frozen Day 3 early dual-arm navigation tuck, one-time physical-column lock,
  locked-column odometry/LiDAR approach, near-shelf lateral-only gate, and safe
  final stand-off.
- Day 4 now intercepts the boundary immediately before Day 3 translation.  Once
  the physical column is locked and both travel arms are ready, the base is held
  stationary while a bounded head scan accumulates independent red, green,
  yellow, and blue RGB/depth/TF tracks in odometry.
- No camera frame must contain all four colours.  Median odometry-frame height
  determines active rows 1–4 from top to bottom.
- The requested row is published before translation, the validated Day 3
  `+0.35 rad` head pose is restored and verified, and the unchanged Day 3
  approach then starts.
- At final stand-off, only the requested colour is reacquired near its locked row
  for synchronized raw-depth geometry and a non-executing one-arm reach screen.
- The old moving-head-during-approach scheduler is disabled.  Close-range
  multi-pose mapping remains recovery-only.
- A separate manually gated position-controlled grasp laboratory remains
  excluded from `solution.launch.py`.

No competition runtime code reads `ERC_SEED`, Gazebo entity names, randomized
spawn order, expected layouts, or world state.  Day 3 RGB/depth limits remain
0.20 s maximum skew and 0.55 s freshness.  No artificial effort command
interface is added.

## Implemented through Day 5

- The manually validated Day 4 one-arm grasp is now executed autonomously from
  the same live world state.
- Public position-only gripper topics remain mandatory; no raw or effort command
  is used.
- The validated 0.10 m extraction and 0.02 m lift are preserved.
- Post-lift retention requires recent fingertip contact, a closed gripper state,
  zero premature fingertip contacts and zero non-fingertip arm contacts.

## Implemented through Day 6

- A stable initial odometry pose is recorded before mission motion.
- The retained book is included in the base-rotation swept-radius calculation.
- The robot retreats straight away from the shelf, verifies clearance, and
  returns to a safe point inside the recorded Start/End Zone.
- Front and rear LiDAR, odometry, contact monitoring and continuous public
  position hold remain active throughout the carry.
