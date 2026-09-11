#!/usr/bin/env python3
"""Validation-only ERC seed oracle; never imported by competition code."""

from __future__ import annotations

import argparse
import json
import random


def expected(seed: int, target_marker: int, colour: str) -> dict:
    random.seed(int(seed))
    colours = ['red', 'green', 'yellow', 'blue']
    book_rows = []
    jitters = []
    for _ in range(5):
        order = colours.copy()
        random.shuffle(order)
        book_rows.append(order)
        jitters.append([random.uniform(-0.25, 0.25) for _ in range(4)])
    markers = list(range(1, 6))
    random.shuffle(markers)
    physical_index = markers.index(target_marker)
    return {
        'seed': int(seed),
        'marker_labels_left_to_right': markers,
        'book_colours_top_to_bottom_by_physical_column': book_rows,
        'target_marker': int(target_marker),
        'target_colour': colour,
        'target_physical_column_left_to_right': physical_index + 1,
        'expected_target_row_top_to_bottom': (
            book_rows[physical_index].index(colour) + 1),
        'validation_only': True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('target_marker', type=int, choices=range(1, 6))
    parser.add_argument(
        'colour', choices=('red', 'green', 'yellow', 'blue'))
    parser.add_argument('--row-only', action='store_true')
    args = parser.parse_args()
    result = expected(args.seed, args.target_marker, args.colour)
    if args.row_only:
        print(result['expected_target_row_top_to_bottom'])
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
