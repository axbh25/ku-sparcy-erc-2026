# Measured-mesh Day 8 repair (Gazebo only)

## Scope

This is an **additive** entry point for the exact inherited interfaces captured in the reproduction archive. It does not edit Day4, Day5, FAST67, the old Day8 files, YAML, URDF, collision meshes, controllers, or Docker configuration. It bypasses the accumulated experimental placement functions rather than appending another override to them.

**This kit has offline geometry/numerical tests and mocked controller-guard tests, not a completed Gazebo physics trial. It is not a hardware robot controller and does not promise a ten-second physical deposit.**

## What changed

1. Register the actual installed bin collision mesh to live RGB-D. The low front lip is no longer represented by an artificial full-height wall. Model dimensions/local shape are used, never an arena/world pose.
2. Fit the tabletop from RGB-D and include the reviewed table's slab and all four legs in collision screening.
3. Search a bounded forward precision-dock and a reachable book location toward the near side of the opening. The captured case passes with a hypothetical 0.18 m forward dock and bin-local book-centre X = -0.12 m.
4. Keep the gravity-supported arm pose during that dock. The entire nominal dock, pre-place, lowering, opening-envelope and retract path must pass before any motion. After docking, measure again and solve with **zero additional docking allowance** before commanding the arm.
5. Replace repeated numerical finite-difference FK inside IK with cached FK, an analytic Jacobian, and active joint-bound handling. Retain the old 2.5 mm / 0.025 rad endpoint acceptance tolerances.
6. Use a nominal floor-contact target only 0.2 mm into the fitted floor instead of 2 mm. This is less penetration, not a relaxed collision limit. Release still requires actual persistent support; a predicted floor crossing never counts as contact.
7. Plan the empty-hand withdrawal to the already screened pre-place height (minimum 0.23 m rise here), verify at least 20 mm hand clearance above the deposited book, then withdraw toward the robot. Replan it from actual open-hand joints after release.

The mesh file digests and inherited source digests are checked. An unreviewed mesh or modified source fails closed. A special internal baseline handles the two coupled fingertip/outer-finger linkage pairs as the aperture changes; contacts with the bin, table, book or other robot links are not whitelisted.

## Offline results

For the supplied scene, the complete nominal plan takes about one second on the analysis machine, excluding about 0.3 seconds of mesh registration. A separate four-times-denser check examines 767 arm-path samples and finds no collision under this model. The original solver/scene failures can be reproduced; the repaired checker still rejects an intentionally placed table intersection. All 20 numerical/replay checks and 10 mocked state/command-guard checks pass.

These are **sampled inflated-envelope** checks, not a continuous collision certificate. The model replay does not test friction, tracking error, actual contact support, release dynamics, simulated real-time factor, or the user's CPU performance. The later RGB-D capture contains zero detected blue pixels; it supplies environment geometry, **not proof of a currently retained book**. Runtime requires current book and contact evidence.

## Install inside erc_sim

After unzipping the kit:

```
python3 /tmp/day8_mesh_repair/install.py --team /opt/erc_ws/src/ku_sparcy_erc
```

This installs only `TEAM/tools/day8_mesh_repair/`. The source compatibility check runs first. Reinstalling backs up only the previous additive kit. There is no colcon rebuild or source patch. Do not run the old Day8 executable for this repair; use the new entry point below.

## Reproduce the tests without moving anything

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 /opt/erc_ws/src/ku_sparcy_erc/tools/day8_mesh_repair/test_repair.py \
  --repro-archive /tmp/day8_repro_20260914_125557.tar.gz \
  --scene-archive /tmp/day8_scene_20260914_132552.tar.gz \
  --output /tmp/day8_mesh_offline_results.json

python3 /opt/erc_ws/src/ku_sparcy_erc/tools/day8_mesh_repair/test_runtime_guards.py \
  --team /opt/erc_ws/src/ku_sparcy_erc
```

Neither test script imports a real ROS runtime or publishes commands. The second script uses fake messages and fake state transitions, so it is **not a ROS integration test**.

## First live simulation trial

Start **one fresh** headless simulator in its own terminal using the team's existing cleanup/launch instructions. Keep the viewer in another terminal. Then, in a third container terminal:

```
bash /opt/erc_ws/src/ku_sparcy_erc/tools/day8_mesh_repair/run_trial.sh 5 blue
```

The runner performs the existing Day4/5 and FAST67 stages, validates the handoff, and invokes this new measured-mesh Day8 entry point. The marker remains a requested identity; physical-order transport is unchanged. The runner explicitly authorizes the additional dock, at no more than 0.22 m, 0.04 m/s and 0.04 m/s^2. Existing return/sideways/retreat speeds are not changed.

During docking the arm is held fixed, nonzero lateral/yaw commands are prohibited, competing `/cmd_vel` publishers are rejected, a front-LiDAR hard stop remains active, cross-track/yaw deviations are bounded, and RGB-D held-book checks continue. After docking, all arm-placement states are base-stationary.

The old live support, stepwise opening, release, actual-state retract and deposit checks remain in the inherited state machine. No gripper opening occurs just because planning passed. Current arm drift, geometry inconsistency or lost required inputs aborts the trial.

Because deliberate docking is now part of Day8, the new validator audits that phase separately instead of falsely claiming zero base movement for the entire node. It replaces the old full-rim rectangle predicate with registered-bin/table predicates; it retains the inherited contact/support/release/deposit checks. The original validator by itself is not the new docking contract.

The latest path is saved in:

```
/opt/erc_ws/src/ku_sparcy_erc/day8_results/last_mesh_repair_run.txt
```

The runner prints the run directory even after failure and does not terminate your interactive shell. Do not keep rerunning the robot on failure. The log/result paths are `day8_mesh.log`, `day8_mesh.json`, and `validation.log` in that directory.

## Direct plan-only preview

Only in an unchanged world with a genuinely retained book and matching current Day7 result:

```
python3 TEAM/tools/day8_mesh_repair/day8_mesh_node.py \
  --ros-args --params-file TEAM/config/day8_place.yaml \
  -p use_sim_time:=true -p operation:=plan \
  -p day7_result_path:=CURRENT_DAY7_JSON -p result_path:=NEW_PREVIEW_JSON
```

Substitute actual paths. Plan-only publishes no actuator commands. Execution is a separate, explicit operation. Historical replay JSON files are never command files.

## Timing

The replayed nominal arm times, at the unchanged configured limits, are about 11.5 simulation seconds pre-place, 10.2 seconds lowering and 16.1 seconds retract, plus opening/support checks. CPU planning time is a different quantity. A slowed simulator takes more wall-clock time. No increase in arm/lowering speeds is smuggled into this kit.

Position-only joint trajectory points assume the existing controller's linear interpolation for position-only input. Do not change interpolation configuration without checking the resulting path. Motion accuracy and successful deposit require a live simulation trial; passing compilation or offline replay is not competition acceptance.
