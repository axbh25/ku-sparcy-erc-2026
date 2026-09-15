# Day 8 - stationary support-before-release placement

Baseline: `day7-stable` = `cb284b8f7188ece2d15f0b5461b02ffb686bd19b`.

## Scope and preservation

Day 5-7 transport, compact carry, book HSV, joint limits, public gripper interfaces,
and all existing planners and validators are unchanged. The original
`pipeline_launch_support.py` and every Day 5-7 launch remain untouched.
`day8_pipeline_support.py` adds a final process after the identical Day 7 chain.
No new external package is required; package.xml therefore remains unchanged.

## Legal sensing and geometry

Day 8 validates the passed Day 7 endpoint and selected Day 5 arm. It reacquires
bin geometry using RGB, raw depth, both CameraInfo models, TF, and current joint
states. Four observed rim edges are required. The installed bin collision STL is
used ONLY for local dimensions, inner clearances and floor-to-rim depth. No world
SDF, spawn order, object pose service, model name lookup or seed is read.

The floor is either a visible depth plane or explicitly recorded as a live rim
height minus local official mesh depth. The latter is an inference, not a claim
that the camera observed the floor. Persistent live support is mandatory in both
cases before release. A partial/occluded rim, unsupported mesh geometry, or
inconsistent floor measurement fails without moving the arm.

The attached 0.16 x 0.02 x 0.25 m (depth/width/height) book box comes from the
measured Day 4 spine and the actual Day 5 grasp FK, then transforms with the
retained gripper. Fresh matching colour/depth points must support this attachment.

## Planning

The installed URDF and existing solve_ik/deterministic_seeds/interpolation helpers
are reused. The grasp orientation is preserved. Three bounded interior targets
are tried; no constraint is relaxed when a candidate fails. PRE_PLACE first
raises clear, then moves above the opening. LOWER_PLACE keeps XY fixed and uses
short predominantly vertical segments at 0.018 m/s. RETRACT first lifts the empty
hand above the bin and deposited book, then moves away.

Collision screening uses inflated collision-mesh OBBs, the bin shell/floor, the
table plane, other robot links and the carried/deposited book. This is a sampled
conservative envelope screen, not a full continuous mesh-collision certificate.
Known initial housing-box overlaps may not grow. A failed screen is not permission
to reduce margins or drive closer. Plan-only must pass on the real installed URDF.

## Contact policy

An opaque held-object collision identity is learned from selected-fingertip
contact. It is never decoded into a shelf, row, colour, seed, or object pose.

| Contact | Before lowering | During lowering | After intended release |
|---|---|---|---|
| Bin/table | Ignore | Ignore | Ignore |
| Robot/bin | Fail | Fail | Fail |
| Same held object/bin | Fail | Support evidence, if footprint/floor test passes | Persistent deposit evidence |
| Different object/bin | Fail | Fail | Fail |
| Selected fingertip loss | Fail | Fail before opening | Expected |

Both contact streams must remain live. Repeated duplicate timestamps do not count
as independent observations. Lowering stops on valid interior support. Penetration
above 6 mm fails. The controller does not drive through a contact to force the
nominal endpoint. It requires settled joints and sustained support before opening.

Release uses public positions 0.018, 0.026, 0.034, 0.040 m, at least 0.8 s each.
No effort field is read or commanded. Hold messages do not overwrite opening while
a slow opening segment is running. Retraction is re-screened in full from the
actual contact-stop joints against the deposited book.

After retraction, the same object must remain in bin contact for at least 2 s;
the open gripper and quiet fingertip evidence are independent checks. Non-red
books also need fresh colour/depth points inside the opening. A red-on-red target
cannot be reliably segmented from the red bin: this limitation is recorded and
the exact contact identity, in-bin pre-release geometry and post-retraction support
are used instead. This does not turn a single bin contact into success.

## Modes

- `survey`: stationary zero base, public position hold, safe head-only viewpoints.
- `plan`: no actuator publishers; live geometry and all three screened paths.
- `execute`: fresh geometry and a fresh plan, then manual stage gates or autonomous
  execution. The normal Day 8 pipeline sets manual_approval_required=false.

A stopped/restarted simulator invalidates old world-dependent result files. Run a
fresh Day 4-7 chain unless the original passed Day 7 world is still live. Pose,
clock, arm and live object checks help reject stale input but are not a universal
world-session identifier.

## Optimized transport handoff tomorrow

Placement reads no route/path history. A different transport can emit the same
passed placement-ready endpoint contract: selected arm, retained Day 5 grasp link,
current compact carry joints, live bin lock, current odometry/yaw/time and zero
premature contacts. No 180-degree world heading is assumed by placement. Keep
RETREAT/ROTATE/LATERAL_TRANSIT/FINAL_ALIGN outside this node; all Day 8 motion stays
arm-only with a stationary base.

## Validation status of the delivered kit

Static parsing, synthetic contact/geometry tests and synthetic Cartesian-chain IK
were run during generation. Live TIAGo/Gazebo placement has NOT been executed by
the assistant. The real-URDF plan-only, isolated motion and fresh full pipeline
gates in the runbook must demonstrate acceptance locally.
