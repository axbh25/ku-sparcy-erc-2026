"""Measured-mesh Day 8 planner. Pure NumPy/OpenCV; no ROS or motion commands.

Local model files supply SHAPE only. Poses are measured with RGB-D/TF.
Screening uses inflated robot mesh bounds against the actual environment mesh,
with discrete joint-space samples. It is not a continuous collision proof or
verification of grasp friction, tracking accuracy, contact support, or release.
"""
from __future__ import annotations
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import cv2
from ku_sparcy_erc.urdf_kinematics import (
    URDFKinematicModel, ChainState, rpy_matrix, homogeneous, rotation_vector, rotation_z,
)
from ku_sparcy_erc.day8_geometry import (
    GeometryError, transform, apply, box_corners, stl_triangles,
    official_bin_profile, fit_bin, red_mask, masked_cloud,
)
from ku_sparcy_erc.day8_planner import CollisionGeometry, sat_gap

VERSION = 'measured_mesh_repair_1'
BIN_SHA = '7c036fe096a50eadc8c32f30837964c3bbbb012cf7a45b874eb7a6f96973aa2a'
TABLE_SHA = '9a662650c305d09e67d112b1c3311c78c8b75454eb05c8c7981292eb1d354977'
BOOK_HALF = np.array([.08, .01, .125])


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise GeometryError('REPAIR_COMPUTE_BUDGET_EXCEEDED')


class FastModel(URDFKinematicModel):
    """The captured URDF model with constant origin matrices cached."""
    def __init__(self, source):
        super().__init__(source.joints_by_name.values())
        self.origins = {n: homogeneous(rpy_matrix(j.origin_rpy), j.origin_xyz)
                        for n, j in self.joints_by_name.items()}

    def forward(self, chain, joint_positions, root_transform=None):
        T = np.eye(4) if root_transform is None else np.array(root_transform).copy()
        links, joints = {}, {}
        if chain:
            links[chain[0].parent] = T.copy()
        for joint in chain:
            T = T @ self.origins[joint.name]
            joints[joint.name] = T.copy()
            value = float(joint_positions.get(joint.name, 0.))
            if joint.movable:
                if not math.isfinite(value) or not joint.lower-1e-7 <= value <= joint.upper+1e-7:
                    raise GeometryError('JOINT_LIMIT:' + joint.name)
                T = T @ self._motion_transform(joint, value)
            links[joint.child] = T.copy()
        return ChainState(T, links, joints)

    def tip_jacobian(self, chain, names, q, fixed):
        values = dict(fixed); values.update(zip(names, q))
        state = self.forward(chain, values)
        T = state.tip_transform
        J = np.zeros((12, len(names)))
        for k, name in enumerate(names):
            joint = self.joints_by_name[name]
            M = state.joint_transforms[name]
            axis = M[:3, :3] @ np.asarray(joint.axis_xyz)
            if joint.joint_type == 'prismatic':
                J[:3, k] = axis
            else:
                J[:3, k] = np.cross(axis, T[:3, 3]-M[:3, 3])
                x, y, z = axis
                cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                J[3:, k] = (.25/math.sqrt(2)*(cross @ T[:3, :3])).ravel()
        return T, J


def solve_endpoint(model, chain, names, fixed, target, rotation, seed, deadline):
    """Analytic Jacobian, damped least squares, active bounds, and line search."""
    lower, upper = model.limits(names, margin=.025)
    q = np.clip(np.asarray(seed, float), lower, upper)
    lam = .002
    for iteration in range(140):
        deadline_check(deadline)
        T, J = model.tip_jacobian(chain, names, q, fixed)
        residual = np.r_[T[:3, 3]-target, (.25/math.sqrt(2)*(T[:3, :3]-rotation)).ravel()]
        pe = float(np.linalg.norm(residual[:3]))
        re = float(np.linalg.norm(rotation_vector(rotation @ T[:3, :3].T)))
        if pe < .0005 and re < .005:
            return q, pe, re, iteration+1
        dq = -np.linalg.solve(J.T @ J + lam**2*np.eye(len(q)), J.T @ residual)
        active = np.ones(len(q), dtype=bool)
        for _ in range(len(q)):
            blocked = ((q-lower < 1e-7) & (dq < 0)) | ((upper-q < 1e-7) & (dq > 0))
            if not np.any(blocked & active):
                break
            active &= ~blocked; dq[:] = 0.
            if not active.any():
                break
            A = J[:, active]
            dq[active] = -np.linalg.solve(A.T @ A + lam**2*np.eye(active.sum()), A.T @ residual)
        dq *= min(1., .12/(float(np.max(np.abs(dq)))+1e-20))
        old_cost = float(residual @ residual)
        improved = False
        for alpha in (1., .5, .25, .125, .0625, .03125):
            deadline_check(deadline)
            trial = np.clip(q+alpha*dq, lower, upper)
            values = dict(fixed); values.update(zip(names, trial))
            U = model.forward(chain, values).tip_transform
            e = np.r_[U[:3, 3]-target, (.25/math.sqrt(2)*(U[:3, :3]-rotation)).ravel()]
            if float(e @ e) < old_cost-1e-14:
                q, lam, improved = trial, max(.0005, lam/1.3), True
                break
        if not improved:
            lam *= 4.
            if lam > 2.:
                break
    values = dict(fixed); values.update(zip(names, q))
    T = model.forward(chain, values).tip_transform
    pe = float(np.linalg.norm(T[:3, 3]-target))
    re = float(np.linalg.norm(rotation_vector(rotation @ T[:3, :3].T)))
    if pe <= .0025 and re <= .025:
        return q, pe, re, iteration+1
    raise GeometryError(f'REPAIR_IK_FAILED:position_error_m={pe:.6f};orientation_error_rad={re:.6f}')


def closest_triangles(points, triangles):
    """Exact closest point to each triangle; small, bounded clouds."""
    out, ids, ds = [], [], []
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    den = d00*d11-d01*d01
    for pp in np.array_split(points, max(1, math.ceil(len(points)/160))):
        ap = pp[:, None, :]-a[None, :, :]
        d20, d21 = (ap*ab).sum(2), (ap*ac).sum(2)
        v = (d11*d20-d01*d21)/np.maximum(den, 1e-24)
        w = (d00*d21-d01*d20)/np.maximum(den, 1e-24)
        face = a[None, :, :] + v[:, :, None]*ab + w[:, :, None]*ac
        valid = (v >= 0)&(w >= 0)&(v+w <= 1)&(den[None, :] > 1e-20)
        best = face.copy()
        dist = np.sum((best-pp[:, None, :])**2, axis=2); dist[~valid] = np.inf
        for start, end in ((a, b), (b, c), (c, a)):
            e = end-start
            t = np.sum((pp[:, None, :]-start)*e, axis=2)/np.maximum((e*e).sum(1), 1e-24)
            q = start + np.clip(t, 0., 1.)[:, :, None]*e
            dd = np.sum((q-pp[:, None, :])**2, axis=2)
            use = dd < dist; best[use], dist[use] = q[use], dd[use]
        which = np.argmin(dist, axis=1); ix = np.arange(len(pp))
        out.append(best[ix, which]); ids.append(which); ds.append(np.sqrt(dist[ix, which]))
    return np.concatenate(out), np.concatenate(ids), np.concatenate(ds)


def triangle_box_intersects(T, half, triangles):
    """Triangle/OBB SAT, including touching, after axis-aligned broad rejection."""
    v = (triangles-T[:3, 3]) @ T[:3, :3]
    v = v[(v.min(1) <= half).all(1) & (v.max(1) >= -half).all(1)]
    if not len(v):
        return False
    edges = np.roll(v, -1, axis=1)-v
    normals = np.cross(edges[:, 0], edges[:, 1])
    axes = np.concatenate([normals[:, None, :],
            np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :]).reshape(-1, 9, 3)], axis=1)
    proj = np.einsum('nvc,nac->nav', v, axes)
    radius = np.einsum('nac,c->na', abs(axes), half)
    separate = (proj.min(2) > radius+1e-10)|(proj.max(2) < -radius-1e-10)
    return bool((~separate.any(1)).any())


def point_inside_mesh(point, triangles):
    """Odd-even ray test supplements surface intersection with containment."""
    d = np.array([1., .371390676, .694746591])
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    h = np.cross(d, e2); det = (e1*h).sum(1); ok = abs(det) > 1e-12
    inv = np.divide(1., det, out=np.zeros_like(det), where=ok)
    s = point-triangles[:, 0]; u = inv*(s*h).sum(1); q = np.cross(s, e1)
    v = inv*(q*d).sum(1); t = inv*(q*e2).sum(1)
    hit = ok & (u >= 0)&(v >= 0)&(u+v <= 1)&(t > 1e-9)
    return bool(len(np.unique(np.round(t[hit], 8))) % 2)


class LocalAssets:
    def __init__(self, description):
        desc = Path(description)
        self.bin_path = desc/'models/collection_bin/meshes/erc_base_collection_bin.STL'
        self.table_path = desc/'models/table/meshes/erc_base_table.STL'
        for path, expected in ((self.bin_path, BIN_SHA), (self.table_path, TABLE_SHA)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise GeometryError('UNREVIEWED_INSTALLED_MESH:' + str(path))
        self.profile = official_bin_profile(self.bin_path)
        C = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        self.bin_tri = stl_triangles(self.bin_path) @ C.T + np.array([.025, 0., -.105])
        n = np.cross(self.bin_tri[:, 1]-self.bin_tri[:, 0], self.bin_tri[:, 2]-self.bin_tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-20)
        self.bin_normal = n
        self.floor_tri = (abs(n[:, 2]) > .99)&(self.bin_tri[:, :, 2].mean(1) < -.15)
        # The reviewed mesh is a slab/legs UNION, not five disconnected parts.
        tt = stl_triangles(self.table_path) @ C.T + np.array([0., 0., -.04])
        xs, ys, zs = [np.unique(np.round(tt[:, :, axis], 6)) for axis in range(3)]
        if (len(xs), len(ys), len(zs)) != (4, 4, 3):
            raise GeometryError('TABLE_MESH_PLANES_NOT_EXPECTED')
        self.table_boxes = []
        def box(lo, hi):
            lo, hi = np.asarray(lo), np.asarray(hi)
            self.table_boxes.append((transform(t=(lo+hi)/2), (hi-lo)/2))
        box([xs[0], ys[0], zs[1]], [xs[-1], ys[-1], zs[-1]])
        for xa, xb in ((xs[0], xs[1]), (xs[2], xs[3])):
            for ya, yb in ((ys[0], ys[1]), (ys[2], ys[3])):
                box([xa, ya, zs[0]], [xb, yb, zs[1]])


def register_bin(points, initial, assets, deadline):
    p = np.asarray(points); initial_B = np.asarray(initial['base_T_bin']); center = initial_B[:3, 3]
    p = p[(np.linalg.norm(p[:, :2]-center[:2], axis=1) < .65)&
          (p[:, 2] > initial['table_z_m']-.025)&(p[:, 2] < center[2]+.03)]
    if len(p) < 150:
        raise GeometryError('MESH_REGISTRATION_TOO_FEW_POINTS')
    p = p[::max(1, len(p)//450)]
    yaw = math.atan2(initial_B[1, 0], initial_B[0, 0]); t = center.copy()
    for _ in range(12):
        deadline_check(deadline)
        R = rotation_z(yaw); local = (p-t) @ R
        closest, ix, distance = closest_triangles(local, assets.bin_tri)
        n = assets.bin_normal[ix] @ R.T; world = closest @ R.T+t
        residual = ((world-p)*n).sum(1)
        dv = np.column_stack((-world[:, 1]+t[1], world[:, 0]-t[0], np.zeros(len(p))))
        J = np.column_stack(((dv*n).sum(1), n))
        weights = np.minimum(1., .004/np.maximum(distance, 1e-12))
        A = J.T @ (weights[:, None]*J)
        if np.linalg.eigvalsh(A)[0] < 1e-5:
            raise GeometryError('BIN_POSE_UNOBSERVABLE_FROM_SURFACES')
        delta = -np.linalg.solve(A+1e-9*np.eye(4), J.T @ (weights*residual))
        delta[0] = np.clip(delta[0], -.03, .03); delta[1:] = np.clip(delta[1:], -.025, .025)
        yaw += delta[0]; t += delta[1:]
        if np.max(abs(delta)) < 1e-6:
            break
    R = rotation_z(yaw); _, _, dist = closest_triangles((p-t) @ R, assets.bin_tri)
    if np.percentile(dist, 80) > .008 or np.median(dist) > .004:
        raise GeometryError('INSTALLED_BIN_MESH_DOES_NOT_MATCH_RGBD')
    if np.linalg.norm(t-center) > .08 or abs(yaw-math.atan2(initial_B[1, 0], initial_B[0, 0])) > .12:
        raise GeometryError('BIN_MESH_REGISTRATION_SHIFT_TOO_LARGE')
    return transform(R, t), {'points': len(p), 'median_m': float(np.median(dist)),
                             'p80_m': float(np.percentile(dist, 80)), 'p95_m': float(np.percentile(dist, 95))}


def fit_table(white_points, scene, prior=None):
    p = np.asarray(white_points)
    expected_z = scene['rim_z_m']-.21
    p = p[(abs(p[:, 2]-expected_z) < .025)&(p[:, 0] > .2)&(p[:, 0] < 2.5)]
    if len(p) < 300:
        raise GeometryError('TABLE_TOP_NOT_OBSERVED')
    edges = np.arange(expected_z-.025, expected_z+.026, .002)
    counts, edges = np.histogram(p[:, 2], bins=edges)
    k = int(np.argmax(counts)); z = (edges[k]+edges[k+1])/2
    p = p[abs(p[:, 2]-z) < .005]; top_z = float(np.median(p[:, 2]))
    if prior is not None:
        M = np.array(prior)
        local = apply(np.linalg.inv(M), p)
        m = (abs(local[:, 0]) <= .415)&(abs(local[:, 1]) <= .715)
        if m.sum() < 250 or np.percentile(abs(p[m, 2]-M[2, 3]), 90) > .008:
            raise GeometryError('TRACKED_TABLE_SURFACE_INCONSISTENT')
        return M, {'points': int(m.sum()), 'top_z_m': top_z, 'pose_source': 'previous_live_RGBD_pose_transformed_by_odometry'}
    rectangle = cv2.boxPoints(cv2.minAreaRect(p[:, :2].astype(np.float32))).astype(float)
    e, f = rectangle[1]-rectangle[0], rectangle[2]-rectangle[1]
    if np.linalg.norm(e) > np.linalg.norm(f): e, f = f, e
    width, length = np.linalg.norm(e), np.linalg.norm(f)
    if abs(width-.8) > .035 or abs(length-1.4) > .045:
        raise GeometryError(f'TABLE_OUTLINE_INCOMPLETE:width={width:.3f};length={length:.3f}')
    ex = e/width
    if ex[0] < 0: ex = -ex
    M = np.eye(4); M[:2, 0] = ex; M[:2, 1] = [-ex[1], ex[0]]
    M[:3, 3] = np.r_[rectangle.mean(0), top_z]
    return M, {'points': len(p), 'top_z_m': top_z, 'observed_size_m': [float(width), float(length)],
               'pose_source': 'live_RGBD_table_top_outline'}


def measured_scene(red_points, white_points, assets, near_xy, prior_table=None, budget=4., initial=None):
    start = time.monotonic(); deadline = start+budget
    scene = fit_bin(red_points, assets.profile, near_xy) if initial is None else dict(initial)
    B, proof = register_bin(red_points, scene, assets, deadline)
    scene.update(base_T_bin=B.tolist(), rim_z_m=float(B[2, 3]),
                 floor_z_m=float(B[2, 3]-.2), outer_half_lengths_m=[.28, .155],
                 floor_source='live_rim_minus_official_mesh_interior_depth', floor_directly_observed=False)
    T, table_proof = fit_table(white_points, scene, prior_table)
    if abs((B[2, 3]-.21)-T[2, 3]) > .008:
        raise GeometryError('BIN_BOTTOM_AND_TABLE_TOP_DISAGREE')
    scene.update(table_z_m=float(T[2, 3]), base_T_table_top=T.tolist(),
                 bin_mesh_registration=proof, table_measurement=table_proof,
                 bin_mesh_sha256=BIN_SHA, table_mesh_sha256=TABLE_SHA,
                 mesh_measurement_wall_sec=time.monotonic()-start, repair_geometry_version=VERSION,
                 pose_initialized_from_odometry=initial is not None, fresh_surface_registration=True)
    return scene


def translate_scene(scene, dx):
    """Hypothetical +X base displacement, not a robot motion command."""
    out = dict(scene)
    for key in ('base_T_bin', 'base_T_table_top'):
        M = np.asarray(scene[key]).copy(); M[0, 3] -= dx; out[key] = M.tolist()
    return out


def sat_many(Ta, ha, Tb, hb):
    """Vectorized equivalent of the captured oriented-box SAT predicate."""
    A, B, d = Ta[:, :3, :3], Tb[:, :3, :3], Tb[:, :3, 3]-Ta[:, :3, 3]
    broad = abs(d)-(np.einsum('nij,nj->ni', abs(A), ha)+np.einsum('nij,nj->ni', abs(B), hb))
    gaps = broad.max(1); use = gaps <= 0.
    if use.any():
        R, S, v = A[use], B[use], d[use]
        axes = np.concatenate((R.transpose(0, 2, 1), S.transpose(0, 2, 1),
                 np.cross(R.transpose(0, 2, 1)[:, :, None, :],
                          S.transpose(0, 2, 1)[:, None, :, :]).reshape(-1, 9, 3)), axis=1)
        norm = np.linalg.norm(axes, axis=2)
        axes /= np.maximum(norm[:, :, None], 1e-20)
        ra = np.einsum('naj,nj->na', abs(np.einsum('nac,ncj->naj', axes, R)), ha[use])
        rb = np.einsum('naj,nj->na', abs(np.einsum('nac,ncj->naj', axes, S)), hb[use])
        exact = abs(np.einsum('nac,nc->na', axes, v))-ra-rb
        exact[norm < 1e-7] = -np.inf
        gaps[use] = exact.max(1)
    return gaps


class MeshScreen:
    """Conservative robot bounds, actual bin shell, measured finite table."""
    def __init__(self, urdf, model, arm, resolve, assets, joints, attach, settings):
        self.model, self.arm, self.assets, self.attach = model, arm, assets, attach
        self.base = CollisionGeometry(urdf, model, arm, resolve)
        self.base.set_start(joints); self.base.set_book_start(joints, attach)
        self.names = [f'arm_{arm}_{i}_joint' for i in range(1, 8)]
        self.chain = model.chain('base_footprint', f'gripper_{arm}_grasping_link')
        self.robot_margin = max(.008, float(settings['robot_margin_m']))
        self.book_margin = max(.012, float(settings['book_margin_m']))
        self.mesh_count = self.base.mesh_count
        self.initial_pairs = self.base.initial_pairs
        # Same-finger four-bar linkage overlaps change as the aperture changes.
        # Use geometry at the current APERTURE, not a closed-hand baseline for
        # open-hand retraction; external collisions remain checked.
        self.aperture_pairs = {}

    def _pairs_at_aperture(self, q):
        value = round(float(q[f'gripper_{self.arm}_finger_joint']), 7)
        if value not in self.aperture_pairs:
            shape = self.poses(q); result = {}
            for side in ('left', 'right'):
                a = f'gripper_{self.arm}_fingertip_{side}_link'
                b = f'gripper_{self.arm}_outer_finger_{side}_link'
                if a in shape and b in shape:
                    result[(a, b)] = min(sat_gap(T, h, U, k) for T, h in shape[a] for U, k in shape[b])
            self.aperture_pairs[value] = result
        return self.aperture_pairs[value]

    def poses(self, joints):
        q = dict(joints)
        for _ in range(8):
            for name, (source, mult, offset) in self.base.mimics.items():
                if source in q: q[name] = q[source]*mult+offset
        cache = {'base_footprint': np.eye(4)}
        def pose(name):
            if name not in cache:
                j = self.model.child_to_joint[name]
                M = pose(j.parent) @ self.model.origins[j.name]
                value = float(q.get(j.name, 0.))
                if j.movable:
                    if not j.lower-1e-7 <= value <= j.upper+1e-7:
                        raise GeometryError('JOINT_LIMIT:'+j.name)
                    M = M @ self.model._motion_transform(j, value)
                cache[name] = M
            return cache[name]
        return {name: [(pose(name) @ M, h) for M, h in boxes]
                for name, boxes in self.base.boxes.items()}

    def check(self, q, scene, carried=True, deposited=None, all_robot=False):
        shapes = self.poses(q); B = np.asarray(scene['base_T_bin']); Ttab = np.asarray(scene['base_T_table_top'])
        tris = apply(B, self.assets.bin_tri.reshape(-1, 3)).reshape(-1, 3, 3)
        trilo, trihi = tris.min((0, 1)), tris.max((0, 1))
        def collides(T, h, triangles=tris, containment=True):
            extent = abs(T[:3, :3]) @ h
            if ((T[:3, 3]+extent < trilo)|(T[:3, 3]-extent > trihi)).any(): return False
            return triangle_box_intersects(T, h, triangles) or (containment and point_inside_mesh(T[:3, 3], tris))
        tables = [(Ttab @ M, h) for M, h in self.assets.table_boxes]
        aperture = self._pairs_at_aperture(q)
        moving = list(shapes) if all_robot else self.base.moving
        extra = .010 if all_robot else 0.0
        violations = []; minimum_table_gap = float('inf')
        flat = [(name, T, h) for name in moving for T, h in shapes[name]]
        Ta = np.array([T for name, T, h in flat]); ha = np.array([h for name, T, h in flat])
        for U, k in tables:
            gaps = sat_many(Ta, ha+self.robot_margin+extra,
                           np.repeat(U[None], len(flat), axis=0), np.repeat(k[None], len(flat), axis=0))
            minimum_table_gap = min(minimum_table_gap, float(gaps.min()))
            for i in np.flatnonzero(gaps < 0): violations.append('ROBOT_TABLE:'+flat[i][0])
        for name, T, h in flat:
            if collides(T, h+self.robot_margin+extra): violations.append('ROBOT_BIN_MESH:'+name)
            if deposited is not None and sat_gap(T, h+.002, deposited, BOOK_HALF+.003) < 0:
                violations.append('ROBOT_DEPOSITED_BOOK:'+name)
        pairs = []
        for a in self.base.moving:
            for b in shapes:
                if a == b or self.base.adjacent(a, b) or (b in self.base.moving and b < a): continue
                baseline = aperture.get((a, b), aperture.get((b, a), self.base.initial_pairs.get((a, b), 0.)))
                for T, h in shapes[a]:
                    for U, k in shapes[b]: pairs.append((a, b, T, h, U, k, baseline))
        if pairs:
            gaps = sat_many(np.array([v[2] for v in pairs]), np.array([v[3] for v in pairs]),
                            np.array([v[4] for v in pairs]), np.array([v[5] for v in pairs]))
            bad = gaps < np.minimum(-.002, np.array([v[6] for v in pairs])-.002)
            for i in np.flatnonzero(bad): violations.append('SELF_ENVELOPE:'+pairs[i][0]+':'+pairs[i][1])
        if carried:
            Tb = self.model.forward(self.chain, q).tip_transform @ self.attach
            h = BOOK_HALF+self.book_margin+extra
            if collides(Tb, h, tris[~self.assets.floor_tri]): violations.append('BOOK_BIN_SHELL')
            raw = apply(Tb, box_corners(-BOOK_HALF, BOOK_HALF))
            if raw[:, 2].min() < scene['floor_z_m']-.006: violations.append('BOOK_FLOOR_PENETRATION')
            for U, k in tables:
                if sat_gap(Tb, BOOK_HALF+np.array([self.book_margin, self.book_margin, .003]), U, k) < 0:
                    violations.append('BOOK_TABLE')
            other = [(name, U, k) for name, body in shapes.items()
                     if not name.startswith(f'gripper_{self.arm}_') for U, k in body]
            gaps = sat_many(np.repeat(Tb[None], len(other), axis=0), np.repeat(h[None], len(other), axis=0),
                            np.array([v[1] for v in other]), np.array([v[2] for v in other]))
            baselines = np.array([self.base.initial_book_pairs.get(v[0], 0.) for v in other])
            for i in np.flatnonzero(gaps < np.minimum(-.002, baselines-self.book_margin-.002)):
                violations.append('BOOK_ROBOT:'+other[i][0])
        return {'passed': not violations, 'violations': sorted(set(violations)),
                'minimum_table_bound_gap_m': minimum_table_gap}


def joint_values(fixed, names, q):
    out = dict(fixed); out.update(zip(names, map(float, q))); return out


def screen_edge(screen, fixed, q0, q1, scene, deadline, carried=True, deposited=None, resolution=.025):
    n = max(2, int(math.ceil(float(np.max(abs(q1-q0)))/resolution))+1)
    worst_gap = math.inf; poses = []
    for f in np.linspace(0., 1., n):
        deadline_check(deadline)
        q = joint_values(fixed, screen.names, q0+f*(q1-q0))
        report = screen.check(q, scene, carried, deposited)
        if not report['passed']: raise GeometryError('MESH_PATH_REJECTED:'+','.join(report['violations']))
        worst_gap = min(worst_gap, report['minimum_table_bound_gap_m'])
        poses.append(screen.model.forward(screen.chain, q).tip_transform[:3, 3])
    # Bound sampled tip speed for positions-only linear joint interpolation.
    tip_derivative = float(np.max(np.linalg.norm(np.diff(poses, axis=0), axis=1))*(n-1))
    return n, worst_gap, tip_derivative


def make_segment(screen, fixed, start_q, goals, rotation, scene, settings, deadline,
                 carried=True, deposited=None, speed=.08, known_first=None):
    model, names, chain = screen.model, screen.names, screen.chain
    q0 = np.asarray(start_q, float); q = q0.copy(); records = []; total = 0.; samples = 0
    max_pe = max_re = max_delta = 0.; min_gap = math.inf
    start_xyz = model.forward(chain, joint_values(fixed, names, q)).tip_transform[:3, 3]
    sequence = []
    if known_first is not None:
        k = np.asarray(known_first, float)
        T = model.forward(chain, joint_values(fixed, names, k)).tip_transform
        err = float(np.linalg.norm(rotation_vector(rotation @ T[:3, :3].T)))
        if err > .025: raise GeometryError('KNOWN_RESTORE_ORIENTATION_MISMATCH')
        sequence.append((k, T[:3, 3], 0., err))
    for goal in goals:
        sequence.append((None, np.asarray(goal, float), None, None))
    for known, target, pe, re in sequence:
        if known is None:
            nxt, pe, re, _ = solve_endpoint(model, chain, names, fixed, target, rotation, q, deadline)
        else: nxt = known
        delta = float(np.max(abs(nxt-q)))
        if delta > 1.5: raise GeometryError('REPAIR_JOINT_BRANCH_JUMP')
        n, gap, derivative = screen_edge(screen, fixed, q, nxt, scene, deadline, carried, deposited,
                                        float(settings.get('joint_sample_rad', .025)))
        duration = max(.10, derivative/speed, delta/float(settings['max_joint_speed_radps']))
        total += duration; samples += n; min_gap = min(min_gap, gap)
        max_pe, max_re, max_delta = max(max_pe, pe), max(max_re, re), max(max_delta, delta)
        records.append(dict(positions=nxt.tolist(), dt_sec=duration, time_sec=total,
                            target_xyz_m=target.tolist(), ik_position_error_m=pe, ik_orientation_error_rad=re))
        q = nxt
    return dict(passed=True, waypoints=records, start_positions_rad=q0.tolist(), start_xyz_m=start_xyz.tolist(),
                target_xyz_m=records[-1]['target_xyz_m'], duration_sec=total,
                max_segment_joint_delta_rad=max_delta, max_total_joint_delta_rad=float(np.max(abs(q-q0))),
                ik_position_error_m=max_pe, ik_orientation_error_rad=max_re,
                collision_screen=dict(passed=True, sampled_mesh_bounds=True, carried_book_screened=carried,
                    deposited_book_screened=not carried, actual_bin_mesh=True, measured_table_mesh=True,
                    samples=samples, max_joint_sample_spacing_rad=float(settings.get('joint_sample_rad', .025)),
                    minimum_table_bound_gap_m=min_gap, not_a_continuous_collision_certificate=True))


def plan_lower(screen, fixed, plan, rotation, scene, settings, deadline):
    q = np.asarray([fixed[n] for n in screen.names])
    T = screen.model.forward(screen.chain, fixed).tip_transform
    target = np.asarray(plan['lower_place_xyz_m'])
    if np.linalg.norm(T[:2, 3]-target[:2]) > .015:
        raise GeometryError('PREPLACE_NOT_LATERALLY_ALIGNED')
    n = max(1, math.ceil(float(np.linalg.norm(target-T[:3, 3]))/.02))
    goals = np.linspace(T[:3, 3], target, n+1)[1:]
    return make_segment(screen, fixed, q, goals, rotation, scene, settings, deadline,
                        speed=float(settings['lower_speed_mps']))


def plan_retract(screen, fixed, plan, rotation, scene, settings, deposited, deadline):
    q = np.asarray([fixed[n] for n in screen.names]); T = screen.model.forward(screen.chain, fixed).tip_transform
    up = T[:3, 3].copy(); up[2] = max(up[2]+.23, plan['pre_place_xyz_m'][2])
    away = up.copy(); away[0] -= float(settings['retract_away_m'])
    goals = []
    for a, b in ((T[:3, 3], up), (up, away)):
        count = max(1, math.ceil(float(np.linalg.norm(b-a))/.02)); goals.extend(np.linspace(a, b, count+1)[1:])
    seg = make_segment(screen, fixed, q, goals, rotation, scene, settings, deadline, carried=False,
                       deposited=deposited, speed=float(settings['retract_speed_mps']))
    up_count = max(1, math.ceil(float(np.linalg.norm(up-T[:3, 3]))/.02))
    at_up = joint_values(fixed, screen.names, seg['waypoints'][up_count-1]['positions'])
    gripper_min = min(apply(M, box_corners(-h-.002, h+.002))[:, 2].min()
                      for name, body in screen.poses(at_up).items()
                      if name.startswith(f'gripper_{screen.arm}_') for M, h in body)
    book_max = apply(deposited, box_corners(-BOOK_HALF, BOOK_HALF))[:, 2].max()
    if gripper_min-book_max < .02:
        raise GeometryError('EMPTY_HAND_NOT_CLEAR_ABOVE_DEPOSIT_BEFORE_WITHDRAWAL')
    return dict(passed=True, segment=seg, samples=seg['collision_screen']['samples'],
                actual_start_positions_rad=q.tolist(), actual_deposited_book_checked=True,
                hand_clearance_before_withdrawal_m=float(gripper_min-book_max),
                new_ik_from_actual_joints=True)


def build_complete_plan(screen, fixed, restore_q, rotation, scene, settings, deadline, offsets=(0., .18, .20, .22)):
    """Search a small bounded docking set; screen the full nominal deposit.

    No motion is commanded. A full successful candidate is required before any
    docking request; after docking the actual geometry must be measured again.
    """
    started = time.monotonic(); failures = []; names = screen.names
    qs = np.asarray([fixed[n] for n in names]); restore_q = np.asarray(restore_q)
    Trestore = screen.model.forward(screen.chain, joint_values(fixed, names, restore_q)).tip_transform
    relative = apply(transform(rotation) @ screen.attach, box_corners(-BOOK_HALF, BOOK_HALF))
    off = rotation @ screen.attach[:3, 3]
    for dx in offsets:
        deadline_check(deadline)
        sc = translate_scene(scene, float(dx)); B = np.asarray(sc['base_T_bin'])
        # Place near, but inside, the robot-facing part of the opening. These are
        # BIN-LOCAL candidate offsets, not arena coordinates or marker numbers.
        for local_x in (-.12, -.13, -.10):
            deadline_check(deadline)
            try:
                center = apply(B, [local_x, 0., 0.])[0]
                pre = np.r_[center[:2]-off[:2], sc['rim_z_m']+float(settings['rim_clearance_m'])-relative[:, 2].min()]
                low = pre.copy(); low[2] = sc['floor_z_m']-.0002-relative[:, 2].min()
                local = apply(np.linalg.inv(B), apply(transform(rotation, low) @ screen.attach,
                                                     box_corners(-BOOK_HALF, BOOK_HALF)))
                half = np.asarray(sc['inner_half_lengths_m'])-.008
                if (abs(local[:, :2]) >= half).any(): raise GeometryError('TARGET_BOOK_OUTSIDE_INNER_OPENING')
                # Cheap reach solve first; do not screen an impossible long path.
                solve_endpoint(screen.model, screen.chain, names, fixed, pre, rotation, restore_q, deadline)
                dock_samples = []
                for part in np.linspace(0., dx, max(2, math.ceil(dx/.01)+1)):
                    deadline_check(deadline)
                    rep = screen.check(fixed, translate_scene(scene, float(part)), all_robot=True)
                    if not rep['passed']: raise GeometryError('DOCK_SWEEP_UNSAFE:'+','.join(rep['violations']))
                    dock_samples.append(float(part))
                preseg = make_segment(screen, fixed, qs, [pre], rotation, sc, settings, deadline,
                                      speed=float(settings['preplace_speed_mps']), known_first=restore_q)
                plan = dict(passed=True, selected_arm=screen.arm, joint_names=names,
                    target_rotation=rotation.tolist(), tip_T_book=screen.attach.tolist(),
                    pre_place_xyz_m=pre.tolist(), lower_place_xyz_m=low.tolist(),
                    retract_xyz_m=(pre+np.array([-.1, 0., .02])).tolist(),
                    segments={'pre_place': preseg}, bin_scene=sc, initial_joint_positions=dict(fixed),
                    collision_meshes_loaded=screen.mesh_count, baseline_housing_envelope_overlaps=len(screen.initial_pairs),
                    book_dimensions_m=[.16,.02,.25], planned_controlled_lowering_m=float(pre[2]-low[2]),
                    book_uncertainty_margin_m=screen.book_margin, repair_version=VERSION,
                    docking_forward_m=float(dx), bin_local_target_x_m=local_x,
                    dock_mesh_sweep_samples_m=dock_samples, source_scene=scene)
                lower_fixed = joint_values(fixed, names, preseg['waypoints'][-1]['positions'])
                lowerseg = plan_lower(screen, lower_fixed, plan, rotation, sc, settings, deadline)
                plan['segments']['lower_place'] = lowerseg
                low_fixed = joint_values(fixed, names, lowerseg['waypoints'][-1]['positions'])
                deposited = screen.model.forward(screen.chain, low_fixed).tip_transform @ screen.attach
                # Screen every planned aperture against the bin/table before release.
                for value in np.linspace(float(fixed[f'gripper_{screen.arm}_finger_joint']), .04, 13):
                    deadline_check(deadline)
                    oq = dict(low_fixed); oq[f'gripper_{screen.arm}_finger_joint'] = float(value)
                    rep = screen.check(oq, sc, carried=False)
                    if not rep['passed']: raise GeometryError('OPENING_ENVIRONMENT_COLLISION:'+','.join(rep['violations']))
                low_fixed[f'gripper_{screen.arm}_finger_joint'] = .04
                retract = plan_retract(screen, low_fixed, plan, rotation, sc, settings, deposited, deadline)
                plan['segments']['retract'] = retract['segment']
                plan['planning_wall_sec'] = time.monotonic()-started
                plan['rejected_candidates'] = failures
                return plan
            except GeometryError as exc:
                failures.append(dict(dock_m=float(dx), local_x_m=local_x, reason=str(exc)))
                if 'BUDGET' in str(exc): raise
    raise GeometryError('NO_FULL_MESH_CHECKED_DEPOSIT:'+str(failures))
