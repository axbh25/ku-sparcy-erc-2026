#!/usr/bin/env python3
"""Numerical one-arm IK and conservative shelf-envelope checks for Day 4.

This is a planning helper, not a simulator shortcut.  It consumes the official
URDF, live joint states, live TF-derived book geometry, and the inherited
front-LiDAR stand-off.  No Gazebo model/entity pose is read.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ku_sparcy_erc.urdf_kinematics import (
    JointSpec,
    URDFKinematicModel,
    rotation_vector,
    rpy_matrix,
)


@dataclass(frozen=True)
class IKSolution:
    success: bool
    joint_names: Tuple[str, ...]
    positions: Tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    seed_index: int
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            'success': self.success,
            'joint_names': list(self.joint_names),
            'positions_rad': [round(float(value), 8) for value in self.positions],
            'position_error_m': round(float(self.position_error_m), 8),
            'orientation_error_rad': round(float(self.orientation_error_rad), 8),
            'iterations': int(self.iterations),
            'seed_index': int(self.seed_index),
            'reason': self.reason,
        }


@dataclass(frozen=True)
class CollisionScreen:
    passed: bool
    minimum_shelf_margin_m: float
    minimum_body_margin_m: float
    maximum_end_effector_x_m: float
    violations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            'passed': self.passed,
            'minimum_shelf_margin_m': round(float(self.minimum_shelf_margin_m), 6),
            'minimum_body_margin_m': round(float(self.minimum_body_margin_m), 6),
            'maximum_end_effector_x_m': round(float(self.maximum_end_effector_x_m), 6),
            'violations': list(self.violations),
        }


@dataclass(frozen=True)
class ArmStagePlan:
    arm: str
    joint_names: Tuple[str, ...]
    target_rpy_base: Tuple[float, float, float]
    pregrasp: IKSolution
    no_contact: IKSolution
    grasp: IKSolution
    extract: IKSolution
    lift: IKSolution
    collision_screens: Mapping[str, CollisionScreen]
    score: float
    passed: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            'arm': self.arm,
            'joint_names': list(self.joint_names),
            'target_rpy_base': [round(float(value), 8) for value in self.target_rpy_base],
            'pregrasp': self.pregrasp.as_dict(),
            'no_contact': self.no_contact.as_dict(),
            'grasp': self.grasp.as_dict(),
            'extract': self.extract.as_dict(),
            'lift': self.lift.as_dict(),
            'collision_screens': {
                name: screen.as_dict()
                for name, screen in self.collision_screens.items()
            },
            'score': round(float(self.score), 8),
            'passed': self.passed,
            'reason': self.reason,
        }


@dataclass(frozen=True)
class GraspPlan:
    selected_arm: str
    candidate_plans: Mapping[str, ArmStagePlan]
    book_surface_xyz_m: Tuple[float, float, float]
    shelf_clearance_m: float
    pregrasp_xyz_m: Tuple[float, float, float]
    no_contact_xyz_m: Tuple[float, float, float]
    grasp_xyz_m: Tuple[float, float, float]
    extract_xyz_m: Tuple[float, float, float]
    lift_xyz_m: Tuple[float, float, float]
    passed: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            'selected_arm': self.selected_arm,
            'candidate_plans': {
                arm: plan.as_dict() for arm, plan in self.candidate_plans.items()
            },
            'book_surface_xyz_m': [round(float(value), 6) for value in self.book_surface_xyz_m],
            'shelf_clearance_m': round(float(self.shelf_clearance_m), 6),
            'pregrasp_xyz_m': [round(float(value), 6) for value in self.pregrasp_xyz_m],
            'no_contact_xyz_m': [round(float(value), 6) for value in self.no_contact_xyz_m],
            'grasp_xyz_m': [round(float(value), 6) for value in self.grasp_xyz_m],
            'extract_xyz_m': [round(float(value), 6) for value in self.extract_xyz_m],
            'lift_xyz_m': [round(float(value), 6) for value in self.lift_xyz_m],
            'passed': self.passed,
            'reason': self.reason,
        }


def pose_error(
    current_transform: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    orientation_weight: float,
) -> Tuple[np.ndarray, float, float]:
    position_error = target_position - current_transform[:3, 3]
    orientation_error = rotation_vector(target_rotation @ current_transform[:3, :3].T)
    return (
        np.concatenate([position_error, float(orientation_weight) * orientation_error]),
        float(np.linalg.norm(position_error)),
        float(np.linalg.norm(orientation_error)),
    )


def solve_ik(
    model: URDFKinematicModel,
    chain: Sequence[JointSpec],
    active_joint_names: Sequence[str],
    fixed_positions: Mapping[str, float],
    target_position_xyz: Sequence[float],
    target_rotation: np.ndarray,
    seed: Sequence[float],
    *,
    seed_index: int = 0,
    max_iterations: int = 220,
    position_tolerance_m: float = 0.018,
    orientation_tolerance_rad: float = 0.16,
    orientation_weight: float = 0.35,
    damping: float = 0.07,
    finite_difference: float = 2.0e-4,
    maximum_step_rad: float = 0.10,
    joint_limit_margin_rad: float = 0.025,
) -> IKSolution:
    names = tuple(str(name) for name in active_joint_names)
    target_position = np.asarray(target_position_xyz, dtype=np.float64)
    target_rotation = np.asarray(target_rotation, dtype=np.float64)
    q = np.asarray(seed, dtype=np.float64).copy()
    lower, upper = model.limits(names, margin=joint_limit_margin_rad)
    if q.shape != lower.shape:
        raise ValueError('IK seed has wrong dimension')
    q = np.clip(q, lower, upper)
    best = (math.inf, math.inf, q.copy(), 0)

    def state_for(values: np.ndarray):
        positions = dict(fixed_positions)
        positions.update({name: float(value) for name, value in zip(names, values)})
        return model.forward(chain, positions)

    for iteration in range(1, max_iterations + 1):
        state = state_for(q)
        error, pos_norm, rot_norm = pose_error(
            state.tip_transform, target_position, target_rotation, orientation_weight)
        metric = pos_norm + 0.18 * rot_norm
        if metric < best[0] + 0.18 * best[1]:
            best = (pos_norm, rot_norm, q.copy(), iteration)
        if pos_norm <= position_tolerance_m and rot_norm <= orientation_tolerance_rad:
            return IKSolution(
                True, names, tuple(float(value) for value in q), pos_norm,
                rot_norm, iteration, seed_index, 'converged')

        jacobian = np.zeros((6, len(names)), dtype=np.float64)
        current = state.tip_transform
        for column in range(len(names)):
            perturbed = q.copy()
            perturbed[column] = min(upper[column], perturbed[column] + finite_difference)
            actual_step = perturbed[column] - q[column]
            if abs(actual_step) < 1.0e-12:
                perturbed[column] = max(lower[column], q[column] - finite_difference)
                actual_step = perturbed[column] - q[column]
            if abs(actual_step) < 1.0e-12:
                continue
            plus = state_for(perturbed).tip_transform
            jacobian[:3, column] = (
                plus[:3, 3] - current[:3, 3]) / actual_step
            differential_rotation = plus[:3, :3] @ current[:3, :3].T
            jacobian[3:, column] = (
                float(orientation_weight)
                * rotation_vector(differential_rotation)
                / actual_step
            )

        jj_t = jacobian @ jacobian.T
        try:
            delta = jacobian.T @ np.linalg.solve(
                jj_t + (float(damping) ** 2) * np.eye(6), error)
        except np.linalg.LinAlgError:
            break
        maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
        if maximum > maximum_step_rad:
            delta *= float(maximum_step_rad) / maximum
        q = np.clip(q + delta, lower, upper)

    pos_norm, rot_norm, best_q, best_iteration = best
    return IKSolution(
        False, names, tuple(float(value) for value in best_q), float(pos_norm),
        float(rot_norm), int(best_iteration), seed_index,
        'did not converge within bounded iterations')


def deterministic_seeds(
    current: Sequence[float],
    lower: np.ndarray,
    upper: np.ndarray,
) -> List[np.ndarray]:
    base = np.clip(np.asarray(current, dtype=np.float64), lower, upper)
    seeds = [base]
    offsets = (
        (0.18, -0.20, 0.18, 0.12, 0.0, 0.12, 0.0),
        (-0.18, -0.12, -0.18, -0.12, 0.0, -0.12, 0.0),
        (0.0, 0.22, 0.0, -0.22, 0.16, 0.0, -0.16),
        (0.0, -0.24, 0.0, 0.18, -0.16, 0.0, 0.16),
    )
    for offset in offsets:
        values = base + np.asarray(offset[:len(base)], dtype=np.float64)
        seeds.append(np.clip(values, lower, upper))
    return seeds


def solve_with_seeds(
    model: URDFKinematicModel,
    chain: Sequence[JointSpec],
    active_joint_names: Sequence[str],
    fixed_positions: Mapping[str, float],
    target_position_xyz: Sequence[float],
    target_rotation: np.ndarray,
    seeds: Sequence[Sequence[float]],
) -> IKSolution:
    results = [
        solve_ik(
            model, chain, active_joint_names, fixed_positions,
            target_position_xyz, target_rotation, seed,
            seed_index=index)
        for index, seed in enumerate(seeds)
    ]
    successful = [result for result in results if result.success]
    pool = successful or results
    return min(
        pool,
        key=lambda result: (
            0 if result.success else 1,
            result.position_error_m + 0.18 * result.orientation_error_rad,
            result.iterations,
        ),
    )


def interpolate_joint_path(
    start: Sequence[float], end: Sequence[float], samples: int = 24,
) -> List[np.ndarray]:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    if start_array.shape != end_array.shape:
        raise ValueError('joint path endpoints have different dimensions')
    if samples < 2:
        raise ValueError('joint path requires at least two samples')
    return [
        (1.0 - alpha) * start_array + alpha * end_array
        for alpha in np.linspace(0.0, 1.0, samples)
    ]


def screen_joint_path(
    model: URDFKinematicModel,
    chain: Sequence[JointSpec],
    active_joint_names: Sequence[str],
    fixed_positions: Mapping[str, float],
    path: Sequence[Sequence[float]],
    *,
    shelf_plane_x_m: float,
    book_surface_xyz_m: Sequence[float],
    stage: str,
    non_gripper_shelf_margin_m: float = 0.055,
    body_keepout_radius_m: float = 0.24,
    body_keepout_x_max_m: float = 0.34,
    selected_column_half_width_m: float = 0.38,
) -> CollisionScreen:
    names = tuple(active_joint_names)
    surface = np.asarray(book_surface_xyz_m, dtype=np.float64)
    violations: List[str] = []
    minimum_shelf_margin = math.inf
    minimum_body_margin = math.inf
    maximum_tip_x = -math.inf

    # The validated compact travel pose can place a moving arm-link origin
    # inside this deliberately coarse torso cylinder.  During the first
    # pregrasp transition, allow that already-existing overlap to egress,
    # but never allow it to become materially worse and require it to clear
    # before the pregrasp target is reached.
    pregrasp_initial_overlaps = set()
    pregrasp_initial_margins = {}
    pregrasp_cleared_overlaps = set()
    previous_body_margins = {}

    for sample_index, sample in enumerate(path):
        positions = dict(fixed_positions)
        positions.update({name: float(value) for name, value in zip(names, sample)})
        state = model.forward(chain, positions)
        tip = state.tip_transform[:3, 3]
        maximum_tip_x = max(maximum_tip_x, float(tip[0]))
        if abs(float(tip[1]) - float(surface[1])) > selected_column_half_width_m:
            violations.append(f'{stage}: tip left selected-column envelope at sample {sample_index}')

        # Ignore the root/shoulder attachment points; screen the moving arm link
        # origins against a coarse torso cylinder and the shelf plane.
        active_children = [model.joints_by_name[name].child for name in names]
        points = [
            state.link_transforms[child][:3, 3]
            for child in active_children
            if child in state.link_transforms
        ]
        for point_index, point in enumerate(points):
            radial = math.hypot(float(point[0]), float(point[1]))
            body_margin = radial - body_keepout_radius_m
            if float(point[0]) <= body_keepout_x_max_m:
                minimum_body_margin = min(minimum_body_margin, body_margin)

                if point_index >= 2:
                    normal_limit = -0.015

                    if (
                        stage == 'pregrasp'
                        and sample_index == 0
                        and body_margin < normal_limit
                    ):
                        pregrasp_initial_overlaps.add(point_index)
                        pregrasp_initial_margins[point_index] = body_margin

                    elif (
                        stage == 'pregrasp'
                        and point_index in pregrasp_initial_overlaps
                        and point_index not in pregrasp_cleared_overlaps
                    ):
                        initial_margin = pregrasp_initial_margins[point_index]

                        if body_margin >= normal_limit:
                            pregrasp_cleared_overlaps.add(point_index)
                        elif body_margin < initial_margin - 0.003:
                            violations.append(
                                f'{stage}: existing torso overlap exceeded '
                                f'3 mm bounded egress allowance at sample '
                                f'{sample_index}')
                    elif body_margin < normal_limit:
                        violations.append(
                            f'{stage}: moving arm entered torso keepout '
                            f'at sample {sample_index}')

                previous_body_margins[point_index] = body_margin
            elif (
                stage == 'pregrasp'
                and point_index in pregrasp_initial_overlaps
            ):
                pregrasp_cleared_overlaps.add(point_index)
            # Last two arm joint origins may approach the shelf, but the wrist
            # and proximal links must retain a margin.  The grasping-link tip is
            # treated separately below.
            if point_index < max(0, len(points) - 2):
                margin = float(shelf_plane_x_m) - non_gripper_shelf_margin_m - float(point[0])
                minimum_shelf_margin = min(minimum_shelf_margin, margin)
                if margin < -0.005:
                    violations.append(
                        f'{stage}: proximal arm crossed shelf margin at sample {sample_index}')

        if not (0.25 <= float(tip[2]) <= 1.90):
            violations.append(f'{stage}: tip height outside bounded workspace')
        if stage in {'pregrasp', 'no_contact'} and float(tip[0]) >= float(surface[0]) - 0.015:
            violations.append(f'{stage}: tip crossed no-contact book surface')
        if stage == 'grasp' and float(tip[0]) > float(surface[0]) + 0.065:
            violations.append('grasp: insertion exceeds bounded 65 mm depth')

    if stage == 'pregrasp':
        uncleared = (
            pregrasp_initial_overlaps
            - pregrasp_cleared_overlaps
        )
        for point_index in sorted(uncleared):
            violations.append(
                'pregrasp: existing torso keepout overlap '
                f'did not clear for arm point {point_index}'
            )

    if math.isinf(minimum_shelf_margin):
        minimum_shelf_margin = 0.0
    if math.isinf(minimum_body_margin):
        minimum_body_margin = 0.0
    unique = tuple(dict.fromkeys(violations))
    return CollisionScreen(
        passed=not unique,
        minimum_shelf_margin_m=float(minimum_shelf_margin),
        minimum_body_margin_m=float(minimum_body_margin),
        maximum_end_effector_x_m=float(maximum_tip_x),
        violations=unique,
    )


def plan_arm_sequence(
    model: URDFKinematicModel,
    arm: str,
    joint_positions: Mapping[str, float],
    book_surface_xyz_m: Sequence[float],
    shelf_clearance_m: float,
    *,
    pregrasp_offset_m: float = 0.18,
    no_contact_offset_m: float = 0.070,
    insertion_depth_m: float = 0.020,
    extract_distance_m: float = 0.10,
    lift_distance_m: float = 0.08,
) -> ArmStagePlan:
    arm = str(arm).strip().lower()
    if arm not in {'left', 'right'}:
        raise ValueError('arm must be left or right')
    names = tuple(f'arm_{arm}_{index}_joint' for index in range(1, 8))
    tip = f'gripper_{arm}_grasping_link'
    chain = model.chain('base_footprint', tip)
    current = np.asarray([joint_positions[name] for name in names], dtype=np.float64)
    lower, upper = model.limits(names, margin=0.025)
    seeds = deterministic_seeds(current, lower, upper)
    surface = np.asarray(book_surface_xyz_m, dtype=np.float64)

    targets = {
        'pregrasp': surface + np.array([-pregrasp_offset_m, 0.0, 0.0]),
        'no_contact': surface + np.array([-no_contact_offset_m, 0.0, 0.0]),
        'grasp': surface + np.array([insertion_depth_m, 0.0, 0.0]),
        'extract': surface + np.array([-extract_distance_m, 0.0, 0.0]),
        'lift': surface + np.array([-extract_distance_m - 0.03, 0.0, lift_distance_m]),
    }

    orientation_candidates = (
        (0.0, 0.0, 0.0),
        (math.pi, 0.0, 0.0),
    )
    candidate_sequences = []
    for rpy in orientation_candidates:
        rotation = rpy_matrix(rpy)
        pre = solve_with_seeds(
            model, chain, names, joint_positions, targets['pregrasp'], rotation, seeds)
        if not pre.success:
            candidate_sequences.append((rpy, pre, None, None, None, None))
            continue
        no_contact = solve_with_seeds(
            model, chain, names, joint_positions, targets['no_contact'], rotation,
            [np.asarray(pre.positions)])
        grasp = solve_with_seeds(
            model, chain, names, joint_positions, targets['grasp'], rotation,
            [np.asarray(no_contact.positions)]) if no_contact.success else None
        extract = solve_with_seeds(
            model, chain, names, joint_positions, targets['extract'], rotation,
            [np.asarray(grasp.positions)]) if grasp is not None and grasp.success else None
        lift = solve_with_seeds(
            model, chain, names, joint_positions, targets['lift'], rotation,
            [np.asarray(extract.positions)]) if extract is not None and extract.success else None
        candidate_sequences.append((rpy, pre, no_contact, grasp, extract, lift))

    def sequence_key(item):
        rpy, pre, no_contact, grasp, extract, lift = item
        solutions = [pre, no_contact, grasp, extract, lift]
        success_count = sum(solution is not None and solution.success for solution in solutions)
        error = sum(
            solution.position_error_m + 0.18 * solution.orientation_error_rad
            for solution in solutions if solution is not None
        )
        displacement = float(np.linalg.norm(np.asarray(pre.positions) - current))
        return (-success_count, error, displacement, abs(rpy[0]))

    chosen = min(candidate_sequences, key=sequence_key)
    rpy, pre, no_contact, grasp, extract, lift = chosen
    fallback = IKSolution(
        False, names, tuple(float(value) for value in pre.positions), math.inf,
        math.inf, 0, -1, 'previous stage did not produce a valid seed')
    no_contact = no_contact or fallback
    grasp = grasp or fallback
    extract = extract or fallback
    lift = lift or fallback
    solutions = {
        'pregrasp': pre,
        'no_contact': no_contact,
        'grasp': grasp,
        'extract': extract,
        'lift': lift,
    }

    screens: Dict[str, CollisionScreen] = {}
    stage_starts = {
        'pregrasp': current,
        'no_contact': np.asarray(pre.positions),
        'grasp': np.asarray(no_contact.positions),
        'extract': np.asarray(grasp.positions),
        'lift': np.asarray(extract.positions),
    }
    for stage, solution in solutions.items():
        if not solution.success:
            screens[stage] = CollisionScreen(
                False, 0.0, 0.0, 0.0,
                (f'{stage}: IK solution unavailable',))
            continue
        path = interpolate_joint_path(stage_starts[stage], solution.positions)
        # The live colour/depth point is on the exposed book spine. Use it
        # to construct a conservative local shelf plane a few centimetres
        # behind the visible surface; retain the front-LiDAR clearance as an
        # independent lower bound. No world/model pose is consulted.
        local_shelf_plane_x = max(
            float(shelf_clearance_m),
            float(surface[0]) + 0.04,
        )
        screens[stage] = screen_joint_path(
            model, chain, names, joint_positions, path,
            shelf_plane_x_m=local_shelf_plane_x,
            book_surface_xyz_m=surface,
            stage=stage)

    all_success = all(solution.success for solution in solutions.values())
    all_screens = all(screen.passed for screen in screens.values())
    score = float(np.linalg.norm(np.asarray(pre.positions) - current))
    passed = all_success and all_screens
    reason = (
        'all five IK stages converged and passed conservative envelope screening'
        if passed else
        'one or more IK stages or conservative envelope screens failed'
    )
    return ArmStagePlan(
        arm=arm,
        joint_names=names,
        target_rpy_base=tuple(float(value) for value in rpy),
        pregrasp=pre,
        no_contact=no_contact,
        grasp=grasp,
        extract=extract,
        lift=lift,
        collision_screens=screens,
        score=score,
        passed=passed,
        reason=reason,
    )


def build_grasp_plan(
    model: URDFKinematicModel,
    joint_positions: Mapping[str, float],
    book_surface_xyz_m: Sequence[float],
    shelf_clearance_m: float,
    preferred_arm: Optional[str] = None,
    **kwargs,
) -> GraspPlan:
    surface = tuple(float(value) for value in book_surface_xyz_m)
    candidates = {
        arm: plan_arm_sequence(
            model, arm, joint_positions, surface, shelf_clearance_m, **kwargs)
        for arm in ('left', 'right')
    }
    passed = [plan for plan in candidates.values() if plan.passed]
    preferred = str(preferred_arm or '').lower()
    if passed:
        selected = min(
            passed,
            key=lambda plan: (
                0 if plan.arm == preferred else 1,
                plan.score,
                0 if plan.arm == 'right' else 1,
            ),
        )
        success = True
        reason = f'{selected.arm} arm selected from collision-screened IK candidates'
    else:
        selected = min(
            candidates.values(),
            key=lambda plan: (plan.score, 0 if plan.arm == 'right' else 1),
        )
        success = False
        reason = 'neither arm produced a complete collision-screened IK sequence'

    array = np.asarray(surface, dtype=np.float64)
    pregrasp = array + np.array([-float(kwargs.get('pregrasp_offset_m', 0.18)), 0, 0])
    no_contact = array + np.array([-float(kwargs.get('no_contact_offset_m', 0.055)), 0, 0])
    grasp = array + np.array([float(kwargs.get('insertion_depth_m', 0.035)), 0, 0])
    extract = array + np.array([-float(kwargs.get('extract_distance_m', 0.10)), 0, 0])
    lift = array + np.array([
        -float(kwargs.get('extract_distance_m', 0.10)) - 0.03,
        0,
        float(kwargs.get('lift_distance_m', 0.08)),
    ])
    return GraspPlan(
        selected_arm=selected.arm,
        candidate_plans=candidates,
        book_surface_xyz_m=surface,
        shelf_clearance_m=float(shelf_clearance_m),
        pregrasp_xyz_m=tuple(float(value) for value in pregrasp),
        no_contact_xyz_m=tuple(float(value) for value in no_contact),
        grasp_xyz_m=tuple(float(value) for value in grasp),
        extract_xyz_m=tuple(float(value) for value in extract),
        lift_xyz_m=tuple(float(value) for value in lift),
        passed=success,
        reason=reason,
    )
