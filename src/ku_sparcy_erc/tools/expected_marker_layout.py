#!/usr/bin/env python3
"""Reproduce only the public ERC marker randomization for validation.

This is a test oracle, never imported by the competition solution.  It mirrors
the public random-call order in simulation.launch.py so a seeded test can check
live visual observations against the expected left-to-right marker order.
"""

from __future__ import annotations

import argparse
import json
import random


def expected_layout(seed: int) -> list[int]:
    random.seed(seed)
    colours = ['red', 'green', 'yellow', 'blue']
    for _column in range(5):
        shuffled = list(colours)
        random.shuffle(shuffled)
        for _row in range(4):
            random.uniform(-0.25, 0.25)
    markers = [1, 2, 3, 4, 5]
    random.shuffle(markers)
    return markers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--target', type=int, choices=range(1, 6))
    args = parser.parse_args()

    layout = expected_layout(args.seed)
    if args.json:
        payload = {'seed': args.seed, 'left_to_right': layout}
        if args.target is not None:
            payload['target'] = args.target
            payload['target_column_index_1_based'] = layout.index(args.target) + 1
        print(json.dumps(payload, sort_keys=True))
    elif args.target is not None:
        print(layout.index(args.target) + 1)
    else:
        print(','.join(str(item) for item in layout))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
