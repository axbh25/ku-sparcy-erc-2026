#!/usr/bin/env python3
"""Verify the installed official ERC gripper/book assets after upstream merge.

This tool is read-only. It parses the current source and installed package share,
checks the canonical 2 cm book model and the position-only physical gripper
interface, and never edits the URDF/SDF or controller configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory


BOOK_RELATIVE = Path('models/book/sdf/erc_book.sdf')
URDF_RELATIVE = Path('urdf/tiago_pro.urdf')
MAIN_FINGER_JOINTS = (
    'gripper_left_finger_joint',
    'gripper_right_finger_joint',
)
FINGERTIP_LINKS = (
    'gripper_left_fingertip_left_link',
    'gripper_left_fingertip_right_link',
    'gripper_right_fingertip_left_link',
    'gripper_right_fingertip_right_link',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def float_text(element: ET.Element | None) -> float | None:
    if element is None or element.text is None:
        return None
    return float(element.text.strip())


def parse_book(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    sizes = []
    for size in root.findall('.//geometry/box/size'):
        if size.text:
            sizes.append([float(item) for item in size.text.split()])
    mus = [float_text(item) for item in root.findall('.//friction/ode/mu')]
    mu2s = [float_text(item) for item in root.findall('.//friction/ode/mu2')]
    mass = float_text(root.find('.//inertial/mass'))
    return {
        'path': str(path),
        'sha256': sha256(path),
        'box_sizes_m': sizes,
        'mu': mus,
        'mu2': mu2s,
        'mass_kg': mass,
    }


def parse_urdf(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    joint_elements = {
        joint.attrib.get('name', ''): joint
        for joint in root.findall('joint')
    }

    mimic_efforts = {}
    for name, joint in joint_elements.items():
        if name.startswith('gripper_') and joint.find('mimic') is not None:
            limit = joint.find('limit')
            mimic_efforts[name] = (
                float(limit.attrib['effort'])
                if limit is not None and 'effort' in limit.attrib else None
            )

    main_limits = {}
    for name in MAIN_FINGER_JOINTS:
        joint = joint_elements.get(name)
        limit = joint.find('limit') if joint is not None else None
        main_limits[name] = {
            'lower': float(limit.attrib['lower']) if limit is not None else None,
            'upper': float(limit.attrib['upper']) if limit is not None else None,
            'effort': float(limit.attrib['effort']) if limit is not None else None,
        }

    fingertip_friction = {}
    for gazebo in root.findall('gazebo'):
        reference = gazebo.attrib.get('reference', '')
        if reference not in FINGERTIP_LINKS:
            continue

        # Each fingertip has multiple Gazebo extension blocks.
        # The friction block contains mu1/mu2 while a later contact-sensor
        # block for the same reference contains neither.  Ignore non-friction
        # blocks so they cannot overwrite the valid friction entry.
        mu1_element = gazebo.find('mu1')
        mu2_element = gazebo.find('mu2')

        if mu1_element is None and mu2_element is None:
            continue

        fingertip_friction[reference] = {
            'mu1': float_text(mu1_element),
            'mu2': float_text(mu2_element),
        }

    ros2_control = {}
    for joint in root.findall('.//ros2_control/joint'):
        name = joint.attrib.get('name', '')
        if name not in MAIN_FINGER_JOINTS:
            continue
        ros2_control[name] = {
            'command_interfaces': [
                item.attrib.get('name', '')
                for item in joint.findall('command_interface')
            ],
            'state_interfaces': [
                item.attrib.get('name', '')
                for item in joint.findall('state_interface')
            ],
        }

    return {
        'path': str(path),
        'sha256': sha256(path),
        'mimic_joint_effort_limits': mimic_efforts,
        'main_finger_joint_limits': main_limits,
        'fingertip_friction': fingertip_friction,
        'ros2_control': ros2_control,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-root',
        default='/opt/erc_ws/src/erc_description',
        help='erc_description source package root')
    parser.add_argument('--result', default='')
    args = parser.parse_args()

    source_root = Path(args.source_root)
    installed_root = Path(get_package_share_directory('erc_description'))
    source_book = source_root / BOOK_RELATIVE
    source_urdf = source_root / URDF_RELATIVE
    installed_book = installed_root / BOOK_RELATIVE
    installed_urdf = installed_root / URDF_RELATIVE

    missing = [
        str(path) for path in (
            source_book, source_urdf, installed_book, installed_urdf)
        if not path.is_file()
    ]
    if missing:
        print('[UPDATED SIMULATOR ASSETS][FAIL] missing files:')
        print('\n'.join(missing))
        return 1

    source_book_data = parse_book(source_book)
    installed_book_data = parse_book(installed_book)
    source_urdf_data = parse_urdf(source_urdf)
    installed_urdf_data = parse_urdf(installed_urdf)

    book_size_ok = (
        len(installed_book_data['box_sizes_m']) >= 2
        and all(
            len(size) == 3
            and abs(size[0] - 0.25) < 1.0e-9
            and abs(size[1] - 0.02) < 1.0e-9
            and abs(size[2] - 0.16) < 1.0e-9
            for size in installed_book_data['box_sizes_m']
        )
    )
    book_friction_ok = (
        installed_book_data['mu']
        and installed_book_data['mu2']
        and all(abs(float(value) - 10.0) < 1.0e-9
                for value in installed_book_data['mu'])
        and all(abs(float(value) - 10.0) < 1.0e-9
                for value in installed_book_data['mu2'])
    )
    mimic = installed_urdf_data['mimic_joint_effort_limits']

    expected_mimic_efforts = {
        'gripper_left_inner_finger_left_joint': 40.0,
        'gripper_left_outer_finger_left_joint': 40.0,
        'gripper_left_fingertip_left_joint': 40.0,
        'gripper_left_finger_right_joint': 10.0,
        'gripper_left_inner_finger_right_joint': 40.0,
        'gripper_left_outer_finger_right_joint': 40.0,
        'gripper_left_fingertip_right_joint': 40.0,
        'gripper_right_inner_finger_left_joint': 40.0,
        'gripper_right_outer_finger_left_joint': 40.0,
        'gripper_right_fingertip_left_joint': 40.0,
        'gripper_right_finger_right_joint': 10.0,
        'gripper_right_inner_finger_right_joint': 40.0,
        'gripper_right_outer_finger_right_joint': 40.0,
        'gripper_right_fingertip_right_joint': 40.0,
    }

    mimic_ok = (
        set(mimic) == set(expected_mimic_efforts)
        and all(
            mimic[name] is not None
            and abs(
                float(mimic[name])
                - expected_mimic_efforts[name]
            ) < 1.0e-9
            for name in expected_mimic_efforts
        )
    )
    fingertip = installed_urdf_data['fingertip_friction']
    fingertip_ok = (
        set(fingertip) == set(FINGERTIP_LINKS)
        and all(
            abs(float(values['mu1']) - 2.7) < 1.0e-9
            and abs(float(values['mu2']) - 2.7) < 1.0e-9
            for values in fingertip.values()
        )
    )
    interfaces = installed_urdf_data['ros2_control']
    position_only_ok = (
        set(interfaces) == set(MAIN_FINGER_JOINTS)
        and all(
            values['command_interfaces'] == ['position']
            and set(values['state_interfaces']) >= {
                'position', 'velocity', 'effort'}
            for values in interfaces.values()
        )
    )
    source_install_ok = (
        source_book_data['sha256'] == installed_book_data['sha256']
        and source_urdf_data['sha256'] == installed_urdf_data['sha256']
    )

    result = {
        'official_upstream_commit_required': (
            '93554d4f9335b2ee3acb49c6b332611f6ad2a964'),
        'source_root': str(source_root),
        'installed_root': str(installed_root),
        'source_book': source_book_data,
        'installed_book': installed_book_data,
        'source_urdf': source_urdf_data,
        'installed_urdf': installed_urdf_data,
        'checks': {
            'book_size_2cm': book_size_ok,
            'book_friction_10': book_friction_ok,
            'mimic_effort_limit_40': mimic_ok,
            'fingertip_friction_2_7': fingertip_ok,
            'position_command_only': position_only_ok,
            'source_matches_installed': source_install_ok,
        },
    }

    if args.result:
        path = Path(args.result)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(path) + '.tmp')
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        os.replace(temporary, path)

    for label, ok in (
        ('OFFICIAL BOOK 2CM', book_size_ok),
        ('OFFICIAL BOOK FRICTION', book_friction_ok),
        ('OFFICIAL MIMIC LIMITS', mimic_ok),
        ('OFFICIAL FINGERTIP FRICTION', fingertip_ok),
        ('POSITION COMMAND ONLY', position_only_ok),
        ('SOURCE INSTALL MATCH', source_install_ok),
    ):
        print(f"[{label}][{'PASS' if ok else 'FAIL'}]")
    print(json.dumps(result, indent=2, sort_keys=True))

    if all(result['checks'].values()):
        print('[UPDATED SIMULATOR ASSETS][PASS]')
        return 0
    print('[UPDATED SIMULATOR ASSETS][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
