# Official ERC Simulator Baseline

## Historical Day 1–3 baseline

- Upstream repository: `dfl-rlab/erc_sim_2026`
- Historical commit: `2aa5a4a7d19177e0bb6625ea5f2c37f7b32bd33a`
- Team Day 3 commit: `9a9768acb824862f2dd4b8de9950ffed54828987`

## Current Day 4 official baseline

- Upstream branch: `main`
- Canonical commit: `93554d4f9335b2ee3acb49c6b332611f6ad2a964`
- Merge: PR #6 from `issue_gripper_effort`
- Supporting final branch commit: `4e0ce63b22e205392e33e732d9dbcf66ce36eaeb`
- README correction: `07f66a3ab8ab60161c2d8aa9f5d985d769d59e10`

Net official source changes from the historical baseline are limited to:

- `src/erc_description/models/book/sdf/erc_book.sdf`
- `src/erc_description/urdf/tiago_pro.urdf`

They provide a 2 cm book spine, increased book/fingertip friction, and increased
mimic-joint effort limits while preserving the physical gripper interface:
position command and position/velocity/effort state.

The update is integrated by a normal merge from `upstream/main`; no historical
team commit is reset or rewritten.
