# `ku_sparcy_erc`

ROS 2 Humble solution package for KU SPARCy's ERC 2026 entry.

## Competition entry point

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

## Revised Day 4 row-perception flow

```text
Day 2 requested marker confirmation
  -> Day 3 locks the physical column once
  -> both validated navigation arms become ready
  -> Day 4 holds the base stationary before translation
  -> bounded, settled head poses observe the selected column
  -> independent RGB/depth/TF colour tracks accumulate in odometry
  -> colour-to-row map locks without a single-frame requirement
  -> requested row is published
  -> head returns to and verifies Day 3 +0.35 rad pose
  -> unchanged Day 3 odometry/LiDAR approach starts
  -> requested colour only is reacquired at final stand-off
  -> live target-book image and synchronized 3-D geometry
  -> non-executing one-arm pre-grasp screen
```

The detector thresholds are unchanged.  Mapping is disabled while Day 3 is
translating; the retired moving-head scheduler fails closed through deprecated
tools.

## Required stationary visibility calibration

The YAML initially contains:

```text
preapproach_scan_calibrated: false
```

Run the selected-column profile before Day 3 translation:

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=5 book_colour:=red \
  day4_stop_after:=preapproach_visibility \
  result_path:=/opt/erc_ws/src/ku_sparcy_erc/day4_preapproach_profile.json
```

Analyze it strictly:

```bash
python3 tools/analyze_day4_preapproach_scan.py \
  day4_preapproach_profile.json \
  --output day4_results/preapproach_analysis.json
```

Apply only the measured minimal pose sequence:

```bash
python3 tools/apply_day4_preapproach_scan.py \
  day4_results/preapproach_analysis.json \
  config/day4_books.yaml
```

The apply tool changes only the calibration flag and competition head sequence;
it verifies that every `book_min_*` detector threshold is byte-for-byte
unchanged.

## Safe Day 4 modes

```bash
# Experimental stationary profile; stops before Day 3 translation
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=5 book_colour:=red \
  day4_stop_after:=preapproach_visibility

# Row mapping, publication, unchanged Day 3 approach and target reacquisition
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=5 book_colour:=red \
  day4_stop_after:=perception

# Add synchronized close-range target-book geometry
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=5 book_colour:=red \
  day4_stop_after:=geometry

# Add non-executing arm reach screen
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=5 book_colour:=red \
  day4_stop_after:=pregrasp
```

## Runtime evidence

The result records pose-wise visibility, settled head state, base drift,
zero/nonzero base commands during scan, raw per-colour observations, spatial
track support, row-lock and first-publication timestamps, Day 3 approach start,
head restoration, fallback use, requested-colour reacquisition, depth/TF book
geometry, and non-executing arm selection.

## Validation

```bash
python3 tools/test_day4_book_perception.py
python3 tools/test_day4_preapproach_tools.py
python3 tools/validate_day4_result.py day4_result.json \
  --target 5 --colour red --expected-row 3 \
  --mode pregrasp --require-preapproach-lock --require-zero-holds
./tools/run_day4_small_regression.sh
python3 tools/summarize_day4_results.py --expected-count 2
```

## Day 5 autonomous pick

```bash
ros2 launch ku_sparcy_erc day5_pipeline.launch.py \
  shelf_column_number:=5 book_colour:=blue
```

The launch records live home odometry, runs the frozen Day 4 pre-grasp path,
executes the tested grasp automatically and verifies post-lift retention.

## Day 6 retained-book return

```bash
ros2 launch ku_sparcy_erc day6_pipeline.launch.py \
  shelf_column_number:=5 book_colour:=blue
```

The robot retreats with the selected arm held at the validated lift pose, checks
the held-book swept radius before rotating and returns to recorded live home
odometry using front/rear LiDAR and continuous position hold.

## Day 7 red-bin approach

```bash
ros2 launch ku_sparcy_erc solution.launch.py \
  shelf_column_number:=5 book_colour:=blue
```

The full competition entry now autonomously picks, returns to the recorded start
zone, detects the red collection bin from RGB-D and stops at a held-book-aware
placement-ready stand-off.  No placement is attempted before Day 8.
