# Day 4 Stationary Pre-Approach Row Mapping

## Empirical reason for the change

A settled-dwell scheduler was tested while Day 3 translated.  The measured
commands were approximately:

```text
14.10 s -> +0.35 rad
15.05 s -> +0.10 rad
20.85 s -> -0.15 rad
20.95 s -> restore +0.35 rad
```

The `+0.35 -> +0.10` transition consumed about 5.8 simulation seconds before a
new settled pose could be commanded.  The `-0.15` pose was cancelled almost
immediately because Day 3 had reached its restore region.  Correct rejection of
unsettled frames then left only three visibility bins.  Day 3 navigation and the
colour detector remained healthy, so changing HSV thresholds would not address
the failure.  The stationary profile therefore tests `-0.10 rad`, close to the
previous useful `-0.106 rad` observation, instead of requiring the cancelled
`-0.15 rad` pose.

## Current state boundary

`Day4Mission` subclasses the frozen `Day3Mission` and overrides only
`_start_approach`.  Day 3 invokes that method after:

1. the physical shelf column is locked;
2. both travel arms have reached the validated PAL home pose; and
3. valid locked-column geometry is available.

Day 4 then holds zero base velocity, scans the selected column, publishes the
requested row, restores `+0.35 rad`, and explicitly calls
`Day3Mission._start_approach(...)`.  No Day 3 source file is edited.

## Spatial row map

Each settled RGB observation is paired with the nearest raw-depth frame under
the inherited limits:

```text
maximum RGB/depth skew: 0.20 simulation seconds
depth stale limit:      0.55 simulation seconds
```

The detected book surface is transformed to odometry.  Red, green, yellow and
blue form independent three-frame tracks across different settled head poses.
No frame needs all four colours.  Median odometry `z` orders active rows from
top to bottom.

## Strict experimental profile

`day4_stop_after:=preapproach_visibility` executes every profile pose, restores
the Day 3 head pose, verifies base drift, and stops before translation.  The
strict analyzer fails unless it can reconstruct a complete spatial map and find
a measured pose subset with adequate support.  It has no inconclusive-to-pass
path and does not modify thresholds.

## Final approach

After calibration, normal operation may end the stationary scan early once all
four tracks lock.  The row is published before translation.  At final stand-off
only the requested colour is reacquired near its locked odometry height; close
multi-row scanning is recovery-only.
