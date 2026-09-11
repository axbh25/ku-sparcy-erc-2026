#!/usr/bin/env python3
"""Apply a passed stationary pre-approach scan recommendation to Day 4 YAML.

Only two fields may change:
  * preapproach_scan_calibrated
  * preapproach_scan_head_tilt_sequence_rad

All HSV/detector thresholds are hashed before and after the atomic edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Dict, List


THRESHOLD_PATTERN = re.compile(
    r'^\s*(book_min_[A-Za-z0-9_]+):\s*(.*?)\s*$', re.MULTILINE
)


def threshold_snapshot(text: str) -> Dict[str, str]:
    return {name: value for name, value in THRESHOLD_PATTERN.findall(text)}


def digest(snapshot: Dict[str, str]) -> str:
    payload = '\n'.join(f'{key}={snapshot[key]}' for key in sorted(snapshot))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'expected exactly one {label}; found {count}')
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()

    try:
        report = json.loads(args.analysis.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[MAPPING SEQUENCE APPLY][FAIL] cannot read analysis: {exc}')
        return 1
    if report.get('status') != 'pass':
        print('[MAPPING SEQUENCE APPLY][FAIL] analysis status is not pass')
        return 1
    if report.get('strategy') != 'stationary_preapproach_scan':
        print('[MAPPING SEQUENCE APPLY][FAIL] wrong analysis strategy')
        return 1
    if report.get('moving_head_during_approach_enabled') is not False:
        print('[MAPPING SEQUENCE APPLY][FAIL] moving-head approach result rejected')
        return 1

    sequence = report.get('recommended_head_tilt_sequence_rad')
    if not isinstance(sequence, list) or not sequence:
        print('[MAPPING SEQUENCE APPLY][FAIL] recommended sequence is empty')
        return 1
    values: List[float] = []
    for value in sequence:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            print('[MAPPING SEQUENCE APPLY][FAIL] non-finite head target')
            return 1
        number = float(value)
        if not -1.0 <= number <= 0.35:
            print('[MAPPING SEQUENCE APPLY][FAIL] head target outside safe range')
            return 1
        values.append(number)

    try:
        original = args.config.read_text(encoding='utf-8')
    except OSError as exc:
        print(f'[MAPPING SEQUENCE APPLY][FAIL] cannot read config: {exc}')
        return 1

    before = threshold_snapshot(original)
    if not before:
        print('[MAPPING SEQUENCE APPLY][FAIL] no detector thresholds found')
        return 1

    formatted = '[' + ', '.join(f'{value:.2f}' for value in values) + ']'
    try:
        updated = replace_once(
            original,
            r'^(\s*preapproach_scan_calibrated:\s*).*$' ,
            r'\1true',
            'preapproach_scan_calibrated field',
        )
        updated = replace_once(
            updated,
            r'^(\s*preapproach_scan_head_tilt_sequence_rad:\s*).*$' ,
            r'\1' + formatted,
            'preapproach_scan_head_tilt_sequence_rad field',
        )
    except ValueError as exc:
        print(f'[MAPPING SEQUENCE APPLY][FAIL] {exc}')
        return 1

    after = threshold_snapshot(updated)
    if before != after:
        print('[MAPPING SEQUENCE APPLY][FAIL] detector thresholds changed')
        return 1

    args.config.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=str(args.config.parent),
        prefix=args.config.name + '.',
        suffix='.tmp',
        delete=False,
    ) as handle:
        handle.write(updated)
        temporary = Path(handle.name)
    temporary.replace(args.config)

    print('[PASS] strategy: stationary_preapproach_scan')
    print('[PASS] moving-head-during-approach remains disabled')
    print('[PASS] recommended sequence: ' + ', '.join(f'{v:+.3f}' for v in values))
    print(f'[PASS] detector-threshold SHA-256 unchanged: {digest(before)}')
    print('[STATIONARY SCAN CALIBRATION APPLY][PASS]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
