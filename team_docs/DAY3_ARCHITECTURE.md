# Day 3 architecture — target-column approach

## Change boundary

`Day3Mission` subclasses the validated Day 2 `MissionStart`. The Day 2 detector,
temporal filter, official result publication, evidence image, complete marker-set
reconciliation, coherent-frame validation reconstruction, and general-search
signed-sweep diagnostic are not edited.

## State extension

```text
Day 2 opening/search states
        |
        v
VERIFYING_EVIDENCE
        |
        v
ESTIMATE_TARGET_GEOMETRY
        |
        +---- stale target ----------> RECOVER_TARGET ----+
        +---- stale depth/LiDAR/TF --> SENSOR_HOLD -------+--> resume safely
        |
        v
APPROACH_TARGET_COLUMN
        |
        +---- hard corridor cluster -> SAFETY_HOLD -------+--> resume or fail
        +---- stale target ----------> RECOVER_TARGET
        +---- stale sensor ----------> SENSOR_HOLD
        |
        v
VERIFY_APPROACH
        |
        v
DONE
```

All holds publish repeated zero-velocity commands. No active-perception pause is
inserted into the validated fast opening turn.

## Frames and signs

- Camera optical: +Z forward, +X right, +Y down.
- `base_footprint`: +X forward, +Y left, +yaw counter-clockwise.
- `/cmd_vel`: `linear.x`, `linear.y`, and `angular.z` use the base convention.
- Lateral control centres the requested column while yaw holds the validated
  post-opening shelf-facing heading; this is the main benefit of the mecanum base.
- TF, not a hard-coded camera or LiDAR mounting pose, performs frame conversion.

## Range fusion

The requested marker's live bounding box selects a padded raw-depth patch. A
foreground-quantile band and median/MAD filter provide a robust marker/shelf
surface depth. The resulting optical point is transformed to `base_footprint`.
Front LaserScan rays are independently transformed to the same frame and
reduced to a swept-body forward corridor.

## Safety

- No translation without fresh target, raw depth, front LiDAR, CameraInfo, and TF.
- RGB/depth timestamp skew must be at most 0.20 simulation seconds.
- LiDAR slowdown begins at 1.50 m.
- A three-point cluster at or inside 0.72 m triggers immediate `SAFETY_HOLD`.
- Automatic release requires at least 0.82 m clearance.
- Desired target/shelf stand-off is 1.05 m.
- Final acceptance also checks lateral and bearing errors for six stable samples.

## Validation progression

1. Pure geometry/controller unit test.
2. Live daemon-independent interface probe.
3. Frozen Day 1 regression.
4. Frozen full Day 2 Phase-1 regression.
5. Motion-disabled ranging trial.
6. Slow 0.45 m approach.
7. Full GUI approach.
8. Two-case headless regression on different targets/layouts.
