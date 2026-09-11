#!/usr/bin/env python3
"""Small dependency-free URDF forward-kinematics model for KU SPARCy.

The module parses the official generated TIAGo Pro URDF at runtime.  It never
modifies that URDF and deliberately uses only Python's standard XML parser and
NumPy, so the submitted solution does not depend on an extra IK package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _numbers(text: Optional[str], expected: int, default: Sequence[float]) -> np.ndarray:
    if text is None:
        values = list(default)
    else:
        values = [float(item) for item in text.split()]
    if len(values) != expected:
        raise ValueError(f'expected {expected} numeric values, got {values}')
    return np.asarray(values, dtype=np.float64)


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw rotation."""
    roll, pitch, yaw = (float(value) for value in rpy)
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one = 1.0 - c
    return np.array([
        [c + x*x*one, x*y*one - z*s, x*z*one + y*s],
        [y*x*one + z*s, c + y*y*one, y*z*one - x*s],
        [z*x*one - y*s, z*y*one + x*s, c + z*z*one],
    ], dtype=np.float64)


def homogeneous(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) rotation vector for a 3x3 matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-9:
        return 0.5 * np.array([
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ], dtype=np.float64)
    if math.pi - angle < 1.0e-5:
        diagonal = np.maximum(0.0, (np.diag(matrix) + 1.0) * 0.5)
        axis = np.sqrt(diagonal)
        if matrix[2, 1] - matrix[1, 2] < 0:
            axis[0] *= -1
        if matrix[0, 2] - matrix[2, 0] < 0:
            axis[1] *= -1
        if matrix[1, 0] - matrix[0, 1] < 0:
            axis[2] *= -1
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-9:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= norm
        return axis * angle
    axis = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=np.float64) / (2.0 * math.sin(angle))
    return axis * angle


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis_xyz: Tuple[float, float, float]
    lower: float
    upper: float

    @property
    def movable(self) -> bool:
        return self.joint_type in {'revolute', 'continuous', 'prismatic'}


@dataclass(frozen=True)
class ChainState:
    tip_transform: np.ndarray
    link_transforms: Mapping[str, np.ndarray]
    joint_transforms: Mapping[str, np.ndarray]


class URDFKinematicModel:
    """Tree model sufficient for one-arm FK and numerical IK."""

    def __init__(self, joints: Iterable[JointSpec]) -> None:
        self.joints_by_name: Dict[str, JointSpec] = {}
        self.child_to_joint: Dict[str, JointSpec] = {}
        for joint in joints:
            if joint.name in self.joints_by_name:
                raise ValueError(f'duplicate joint name: {joint.name}')
            if joint.child in self.child_to_joint:
                raise ValueError(f'link has multiple parents: {joint.child}')
            self.joints_by_name[joint.name] = joint
            self.child_to_joint[joint.child] = joint

    @classmethod
    def from_xml(cls, xml_text: str) -> 'URDFKinematicModel':
        root = ET.fromstring(xml_text)
        joints: List[JointSpec] = []
        for element in root.findall('joint'):
            name = element.attrib.get('name', '').strip()
            joint_type = element.attrib.get('type', 'fixed').strip()
            parent_element = element.find('parent')
            child_element = element.find('child')
            if not name or parent_element is None or child_element is None:
                continue
            parent = parent_element.attrib.get('link', '').strip()
            child = child_element.attrib.get('link', '').strip()
            if not parent or not child:
                continue
            origin = element.find('origin')
            xyz = _numbers(
                origin.attrib.get('xyz') if origin is not None else None,
                3, (0.0, 0.0, 0.0))
            rpy = _numbers(
                origin.attrib.get('rpy') if origin is not None else None,
                3, (0.0, 0.0, 0.0))
            axis_element = element.find('axis')
            axis = _numbers(
                axis_element.attrib.get('xyz') if axis_element is not None else None,
                3, (1.0, 0.0, 0.0))
            limit = element.find('limit')
            if joint_type == 'continuous':
                lower, upper = -math.pi, math.pi
            elif joint_type in {'revolute', 'prismatic'}:
                if limit is None:
                    raise ValueError(f'movable joint {name} has no limits')
                lower = float(limit.attrib.get('lower', '-3.141592653589793'))
                upper = float(limit.attrib.get('upper', '3.141592653589793'))
            else:
                lower = upper = 0.0
            joints.append(JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin_xyz=tuple(float(value) for value in xyz),
                origin_rpy=tuple(float(value) for value in rpy),
                axis_xyz=tuple(float(value) for value in axis),
                lower=lower,
                upper=upper,
            ))
        return cls(joints)

    @classmethod
    def from_file(cls, path: str) -> 'URDFKinematicModel':
        with open(path, 'r', encoding='utf-8') as handle:
            return cls.from_xml(handle.read())

    def chain(self, root_link: str, tip_link: str) -> List[JointSpec]:
        current = str(tip_link)
        reverse: List[JointSpec] = []
        visited = set()
        while current != root_link:
            if current in visited:
                raise ValueError('cycle detected while building URDF chain')
            visited.add(current)
            joint = self.child_to_joint.get(current)
            if joint is None:
                raise ValueError(
                    f'cannot connect {tip_link} to {root_link}; '
                    f'no parent joint for {current}')
            reverse.append(joint)
            current = joint.parent
        return list(reversed(reverse))

    @staticmethod
    def _motion_transform(joint: JointSpec, position: float) -> np.ndarray:
        if joint.joint_type in {'revolute', 'continuous'}:
            return homogeneous(axis_angle_matrix(joint.axis_xyz, position), (0, 0, 0))
        if joint.joint_type == 'prismatic':
            translation = np.asarray(joint.axis_xyz, dtype=np.float64) * float(position)
            return homogeneous(np.eye(3), translation)
        return np.eye(4, dtype=np.float64)

    def forward(
        self,
        chain: Sequence[JointSpec],
        joint_positions: Mapping[str, float],
        root_transform: Optional[np.ndarray] = None,
    ) -> ChainState:
        transform = (
            np.asarray(root_transform, dtype=np.float64).copy()
            if root_transform is not None else np.eye(4, dtype=np.float64)
        )
        if transform.shape != (4, 4):
            raise ValueError('root_transform must be 4x4')
        link_transforms: Dict[str, np.ndarray] = {}
        joint_transforms: Dict[str, np.ndarray] = {}
        if chain:
            link_transforms[chain[0].parent] = transform.copy()
        for joint in chain:
            origin = homogeneous(rpy_matrix(joint.origin_rpy), joint.origin_xyz)
            transform = transform @ origin
            joint_transforms[joint.name] = transform.copy()
            position = float(joint_positions.get(joint.name, 0.0))
            if joint.movable:
                margin = 1.0e-7
                if position < joint.lower - margin or position > joint.upper + margin:
                    raise ValueError(
                        f'{joint.name}={position:.6f} outside '
                        f'[{joint.lower:.6f}, {joint.upper:.6f}]')
            transform = transform @ self._motion_transform(joint, position)
            link_transforms[joint.child] = transform.copy()
        return ChainState(
            tip_transform=transform,
            link_transforms=link_transforms,
            joint_transforms=joint_transforms,
        )

    def limits(self, joint_names: Sequence[str], margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = [], []
        for name in joint_names:
            joint = self.joints_by_name[name]
            lo = float(joint.lower) + float(margin)
            hi = float(joint.upper) - float(margin)
            if lo >= hi:
                lo, hi = float(joint.lower), float(joint.upper)
            lower.append(lo)
            upper.append(hi)
        return np.asarray(lower), np.asarray(upper)
