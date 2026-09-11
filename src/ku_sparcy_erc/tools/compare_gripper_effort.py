#!/usr/bin/env python3
"""Compare free-air and loaded effort-state samples without treating them as control."""

from __future__ import annotations

import argparse
import json
import math


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('free_air_result')
    parser.add_argument('grasp_result')
    args = parser.parse_args()

    with open(args.free_air_result, encoding='utf-8') as handle:
        free = json.load(handle)
    with open(args.grasp_result, encoding='utf-8') as handle:
        grasp = json.load(handle)

    free_stats = free.get('close_effort_free_air') or {}
    loaded_stats = grasp.get('gripper_effort_state_statistics') or {}
    free_mean = free_stats.get('mean')
    loaded_mean = loaded_stats.get('mean')
    free_std = free_stats.get('stdev')
    loaded_std = loaded_stats.get('std')
    available = all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (free_mean, loaded_mean, free_std, loaded_std)
    )
    if not available:
        print('[EFFORT LOAD DIAGNOSTIC][FAIL] usable statistics are missing')
        return 1

    delta = float(loaded_mean) - float(free_mean)
    pooled = math.sqrt(float(free_std) ** 2 + float(loaded_std) ** 2)
    normalized = abs(delta) / max(1.0e-9, pooled)
    print(json.dumps({
        'free_air_close_mean': free_mean,
        'free_air_close_std': free_std,
        'loaded_stage_mean': loaded_mean,
        'loaded_stage_std': loaded_std,
        'mean_delta': delta,
        'delta_over_pooled_std': normalized,
        'used_for_control': False,
    }, indent=2, sort_keys=True))
    print('[EFFORT LOAD DIAGNOSTIC][PASS]')
    if normalized >= 3.0:
        print('[EFFORT LOAD DISCRIMINATION][OBSERVED] repeatability still requires more trials')
    else:
        print('[EFFORT LOAD DISCRIMINATION][NOT RELIABLE] do not use effort as a grasp controller')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
