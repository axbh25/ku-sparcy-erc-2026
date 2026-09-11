# Superseded Day 4 Approach-Time Mapping Experiment

This document records a rejected experiment rather than the competition
architecture.

The experiment attempted to map book rows while the Day 3 base was already
approaching the shelf.  Fixed `+0.35 rad` observations showed several colours,
but a bounded moving-head scheduler created long visibility gaps because the
head required several simulation seconds to settle.  The lower pose was then
cancelled by the final Day 3 head-restore region before it could produce useful
settled frames.

The experiment is permanently retired:

- mapping while Day 3 translates is disabled;
- the moving-head-during-approach scheduler is a no-op;
- old visibility-window tools fail closed;
- detector/HSV thresholds were not changed.

See `DAY4_STATIONARY_PREAPPROACH_REDESIGN.md` for the current architecture.
