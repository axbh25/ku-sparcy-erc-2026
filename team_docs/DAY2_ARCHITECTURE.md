# Day 2 Architecture Record

## State flow

### Phase 1 fast start

`WAITING_FOR_INPUTS -> ROTATING_FAST -> WAITING_FOR_TARGET (only if needed) -> VERIFYING_EVIDENCE -> DONE`

### Orientation-independent mode

`WAITING_FOR_INPUTS -> SEARCH_FOR_SHELF -> ALIGN_TO_TARGET -> VERIFYING_EVIDENCE -> DONE`

Any missing input, simulation-time timeout, excessive search sweep, stale target
without recovery, missing official publication, or failed annotated-image write
ends in a recorded failure and repeated zero-velocity commands.

## Timing policy

- Readiness timeout: wall clock, so a stalled simulator fails.
- Motion/search/detection deadlines: `/clock` simulation time, so a low Gazebo
  real-time factor does not create false failures.
- Result JSON: both simulation and wall timing.

## Perception policy

- Detection starts on the first rate-eligible live RGB frame, including while
  the base is turning.
- A single-frame classification is never a scored result.
- The target is accepted after at least three spatially consistent frames,
  minimum average confidence, and classifier-margin checks.
- The target result is published and the annotated target-column image is saved immediately
  upon confirmation; the competition path does not wait for the other markers.
- All-marker confirmation is test-only and supports seeded regression.

## Day 3 extension point

Day 3 will consume the confirmed target bounding box, camera intrinsics/depth,
and target bearing to produce a safe shelf-column approach while retaining this
perception node's evidence and timeout behavior.
