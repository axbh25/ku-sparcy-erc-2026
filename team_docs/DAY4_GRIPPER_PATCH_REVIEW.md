# Day 4 official gripper-update review

## Canonical official state

Day 4 targets official `dfl-rlab/erc_sim_2026` main at:

```text
93554d4f9335b2ee3acb49c6b332611f6ad2a964
```

This is merge PR #6 from `issue_gripper_effort`. The supporting final branch
commit is `4e0ce63b22e205392e33e732d9dbcf66ce36eaeb`; commit `07f66a3` restores the
README `_raw` controller names after the branch reverts. `8b5f261` is historical
development context, not the canonical standalone target.

## Net official model changes from the validated Day 3 baseline

Only these official source files differ:

- `src/erc_description/models/book/sdf/erc_book.sdf`
- `src/erc_description/urdf/tiago_pro.urdf`

The final net changes are:

- book box size `0.25 0.03 0.16` -> `0.25 0.02 0.16` m;
- book `mu` and `mu2` `5.0` -> `10.0`;
- twelve gripper mimic-joint effort limits `0.1` -> `40.0`;
- four fingertip `mu1` and `mu2` values `0.9` -> `2.7`.

The final merge does not add an effort command interface, alter the public
position-controller architecture, modify Gazebo, or modify `gz_ros2_control`.
The physical-interface model remains position command plus
position/velocity/effort state.

## Uploaded patch evidence

`8b5f261.patch` also contains a binary mesh edit and generated-URDF mesh
reference regressions. `4e0ce63.patch` removes those mesh-reference edits and
retains only the final SDF/URDF text changes above.

The patches are inspected read-only. They are never cherry-picked or applied.
The official merge commit is integrated through a normal Git merge from
`upstream/main`, preserving the validated team branch history.

## Experimental status

The maintainer reports successful book pickup and merged the branch. Local
success is still a required gate. Day 4 first validates perception, range,
position-only interfaces, numerical IK, conservative envelope screening, live
contacts, and each manually approved motion stage before extraction or lift.
