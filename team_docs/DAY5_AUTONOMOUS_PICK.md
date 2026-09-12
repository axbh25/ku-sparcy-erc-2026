# Day 5 — Autonomous retained-book pick

Day 5 converts the manually gated Day 4 grasp into an automatic sequence without
changing any Day 1–4 perception, navigation, IK, collision-screening, gripper,
extraction, or lift constants.  It preserves the public position-only gripper
interface, 0.10 m extraction, and physically validated 0.02 m lift.

After lift settling, the node holds the final public close-position command and
requires a closed joint position, fresh selected-fingertip contact, zero
premature contacts, and zero non-fingertip selected-arm contacts.
