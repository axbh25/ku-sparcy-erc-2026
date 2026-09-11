# KU SPARCy Day 4 architecture

## Production-safe competition entry

```text
Day 3 safe stand-off
  -> selected physical column remains locked
  -> base stop
  -> bounded head scan
  -> HSV/shape detection of all four coloured spines
  -> coherent top-to-bottom row confirmation
  -> /erc/shelf_row_identification (Int32)
  -> live target-book image
  -> nearest-timestamp raw-depth geometry
  -> base_footprint transform
  -> geometric arm selection and pre-grasp screen
  -> stop
```

`solution.launch.py` deliberately stops before new Day 4 arm or gripper motion.
This preserves the stable scoring path while manipulation is developed.

## Explicit staged manipulation laboratory

```text
passed Day 4 geometry result
  -> verify current base odometry still matches result
  -> parse the installed official URDF
  -> numerical IK for both arms
  -> choose one complete collision-envelope-screened sequence
  -> PLAN approval
  -> PREGRASP approval
  -> OPEN public position gripper command
  -> NO-CONTACT approval
  -> bounded open-finger contact pose
  -> separate position closure approval
  -> contact and visual validation
  -> small extraction
  -> small lift
```

Every physical stage waits for `std_msgs/msg/Bool(data=true)` on
`/ku_sparcy/day4_continue`. The base publishes repeated zero velocity. The
unselected arm stays in the Day 3 navigation pose; the torso is never
commanded. The gripper is commanded only on the public clamped position topic.
Effort is recorded as state/diagnostic data only.

The numerical collision screen is intentionally conservative but is not a full
mesh collision checker. The laboratory therefore transforms the front LiDAR
into the Day 3 swept corridor, aborts on clustered close obstacles, rejects
non-fingertip contact by the selected arm, and still requires manual Gazebo
inspection before each stage. During extraction and lift it republishes the
final public position target at 5 Hz; effort remains diagnostic state only. A staged result is not promoted
into the autonomous competition entry until repeated local grasp trials pass.
