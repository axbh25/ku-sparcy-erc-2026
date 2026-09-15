"""Day 8 IK plus carried-book and robot collision-geometry envelope screening."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
from ku_sparcy_erc.grasp_planner import solve_ik, deterministic_seeds, interpolate_joint_path
from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel, rpy_matrix, rotation_vector
from ku_sparcy_erc.day8_geometry import (GeometryError,transform,apply,box_corners,
    stl_triangles,book_bounds_in_bin)


def origin(el):
    if el is None: return np.eye(4)
    return transform(rpy_matrix(list(map(float,el.get('rpy','0 0 0').split()))),
                     list(map(float,el.get('xyz','0 0 0').split())))


def sat_gap(Ta,ha,Tb,hb):
    """Separating-axis gap for two oriented bounding boxes (negative=overlap)."""
    A=Ta[:3,:3]; B=Tb[:3,:3]; d=Tb[:3,3]-Ta[:3,3]
    broad=np.abs(d)-(np.abs(A)@ha+np.abs(B)@hb)
    if np.max(broad)>0:return float(np.max(broad))
    axes=[A[:,i] for i in range(3)]+[B[:,i] for i in range(3)]
    axes += [np.cross(A[:,i],B[:,j]) for i in range(3) for j in range(3)]
    gaps=[]
    for a in axes:
        norm=np.linalg.norm(a)
        if norm<1e-7: continue
        a=a/norm
        gaps.append(abs(d@a)-np.abs(A.T@a)@ha-np.abs(B.T@a)@hb)
    return float(max(gaps))


class CollisionGeometry:
    def __init__(self,urdf_path,model,arm,resolve_package):
        root=ET.parse(urdf_path).getroot();self.model=model;self.arm=arm
        self.boxes={};self.mimics={};self.mesh_count=0
        self.parents={j.child:j.parent for j in model.joints_by_name.values()}
        for j in root.findall('joint'):
            m=j.find('mimic')
            if m is not None:
                self.mimics[j.get('name')]=(m.get('joint'),float(m.get('multiplier','1')),float(m.get('offset','0')))
        for link in root.findall('link'):
            name=link.get('name','')
            if not (name.startswith(('arm_','gripper_','torso_','head_')) or name=='base_link'):
                continue
            boxes=[]
            for collision in link.findall('collision'):
                g=collision.find('geometry')
                if g is None: continue
                mesh=g.find('mesh'); box=g.find('box');sphere=g.find('sphere');cylinder=g.find('cylinder')
                if mesh is not None:
                    uri=mesh.get('filename','')
                    if uri.startswith('package://'):
                        pkg,rel=uri[10:].split('/',1);path=Path(resolve_package(pkg))/rel
                    elif uri.startswith('file://'): path=Path(uri[7:])
                    else: path=Path(uri)
                    v=stl_triangles(path).reshape(-1,3)
                    scale=np.array(list(map(float,mesh.get('scale','1 1 1').split())))
                    v*=scale;lo=v.min(0);hi=v.max(0);self.mesh_count+=1
                elif box is not None:
                    h=np.array(list(map(float,box.get('size').split())))/2;lo=-h;hi=h
                elif sphere is not None:
                    h=np.full(3,float(sphere.get('radius')));lo=-h;hi=h
                elif cylinder is not None:
                    h=np.array([float(cylinder.get('radius'))]*2+[float(cylinder.get('length'))/2]);lo=-h;hi=h
                else: raise GeometryError('UNSUPPORTED_URDF_COLLISION_GEOMETRY: '+name)
                if not np.isfinite(lo).all() or max(hi-lo)>2: raise GeometryError('INVALID_MESH_UNITS: '+name)
                boxes.append((origin(collision.find('origin'))@transform(t=(lo+hi)/2),(hi-lo)/2))
            if boxes:self.boxes[name]=boxes
        self.moving=[n for n in self.boxes if n.startswith((f'arm_{arm}_',f'gripper_{arm}_'))]
        if len(self.moving)<7: raise GeometryError('TOO_FEW_SELECTED_ARM_COLLISION_GEOMETRIES')
        self.chains={n:model.chain('base_footprint',n) for n in self.boxes}
        self.initial_pairs={};self.initial_book_pairs={}

    def poses(self,joints):
        q=dict(joints)
        for _ in range(8):
            for name,(source,mult,offset) in self.mimics.items():
                if source in q:q[name]=q[source]*mult+offset
        out={}
        for name,boxes in self.boxes.items():
            T=self.model.forward(self.chains[name],q).tip_transform
            out[name]=[(T@o,h) for o,h in boxes]
        return out

    def adjacent(self,a,b):
        for child,parent in ((a,b),(b,a)):
            cur=child
            for _ in range(3):
                cur=self.parents.get(cur)
                if cur==parent:return True
        return False

    def set_start(self,q):
        shapes=self.poses(q)
        for a in self.moving:
            for b in shapes:
                if a==b or self.adjacent(a,b):continue
                if b in self.moving and b<a:continue
                gap=min(sat_gap(T,h,U,k) for T,h in shapes[a] for U,k in shapes[b])
                if gap<0:self.initial_pairs[(a,b)]=gap

    def set_book_start(self,q,Tattach):
        tip=self.model.forward(self.model.chain('base_footprint',f'gripper_{self.arm}_grasping_link'),q).tip_transform
        Tb=tip@Tattach
        self.initial_book_pairs={}
        for name,shapes in self.poses(q).items():
            if name.startswith(f'gripper_{self.arm}_'):continue
            gap=min(sat_gap(Tb,np.array([.08,.01,.125]),T,h) for T,h in shapes)
            if gap<0:self.initial_book_pairs[name]=gap

    def check(self,q,scene,Tattach,carried=True,deposited_T=None,margin=.008,book_margin=.012):
        shapes=self.poses(q);B=np.array(scene['base_T_bin']);inv=np.linalg.inv(B)
        hx,hy=scene['inner_half_lengths_m'];ox,oy=scene['outer_half_lengths_m']
        floor=scene['floor_z_m']-scene['rim_z_m'];table=scene['table_z_m']-scene['rim_z_m']
        # Four conservative continuous walls, from table level to observed rim.
        obstacles=[]
        for axis,sign in ((0,-1),(0,1),(1,-1),(1,1)):
            h=np.array([ox,oy,-table/2]);c=np.array([0.,0.,table/2])
            inn=(hx,hy)[axis];out=(ox,oy)[axis]
            h[axis]=(out-inn)/2;c[axis]=sign*(out+inn)/2
            obstacles.append((B@transform(t=c),h))
        minimum=float('inf');violations=[]
        for a in self.moving:
            for T,h in shapes[a]:
                # Mesh AABB is conservative; inflate for geometry/controller error.
                inflated=h+margin
                local=apply(inv@T,box_corners(-inflated,inflated))
                lo=local.min(0)
                hi=local.max(0)

                # The table-plane test is meaningful only where the robot
                # envelope is actually over the table.  The previous global-Z
                # test falsely rejected shoulder/upper-arm links near the
                # robot simply because they were lower than the tabletop.
                #
                # The live bin footprint anchors the table region.  Expand it
                # conservatively beyond the bin on every side; real unexpected
                # robot/table contact remains independently fatal on /contacts.
                table_guard_m = 0.15
                table_hx = ox + table_guard_m
                table_hy = oy + table_guard_m

                overlaps_table_xy = (
                    lo[0] < table_hx
                    and hi[0] > -table_hx
                    and lo[1] < table_hy
                    and hi[1] > -table_hy
                )

                if (
                    overlaps_table_xy
                    and lo[2] < table + .008
                ):
                    violations.append('ARM_TABLE_PLANE:'+a)

                for U,k in obstacles:
                    gap=sat_gap(T,inflated,U,k);minimum=min(minimum,gap)
                    if gap<0:violations.append('ARM_BIN_WALL:'+a)
                # Bin floor only matters when the collision envelope overlaps its footprint.
                if lo[0]<ox and hi[0]>-ox and lo[1]<oy and hi[1]>-oy and lo[2]<floor+.008:
                    violations.append('ARM_BIN_FLOOR:'+a)
                if deposited_T is not None:
                    gap=sat_gap(T,h+.002,deposited_T,np.array([.08,.01,.125])+.003)
                    minimum=min(minimum,gap)
                    if gap<0:violations.append('ARM_DEPOSITED_BOOK:'+a)
            for b in shapes:
                if a==b or self.adjacent(a,b):continue
                if b in self.moving and b<a:continue
                gap=min(sat_gap(T,h,U,k) for T,h in shapes[a] for U,k in shapes[b])
                # Stable housing-box overlaps are conservative modelling artefacts;
                # no pair may penetrate farther than its measured starting bound.
                baseline=self.initial_pairs.get((a,b),0.)
                if gap<min(-.002,baseline-.002):violations.append('SELF_ENVELOPE:'+a+':'+b)
        if carried:
            tip=self.model.forward(self.model.chain('base_footprint',f'gripper_{self.arm}_grasping_link'),q).tip_transform
            Tb=tip@Tattach
            for U,k in obstacles:
                gap=sat_gap(Tb,np.array([.08,.01,.125])+book_margin,U,k)
                minimum=min(minimum,gap)
                if gap<0:violations.append('CARRIED_BOOK_BIN_WALL')
            for name,robot_shapes in shapes.items():
                # Selected grasp-contact links are handled by live tactile
                # identity checks. Other arm/body/head collisions are forbidden.
                if name.startswith(f'gripper_{self.arm}_'):continue
                gap=min(sat_gap(Tb,np.array([.08,.01,.125])+book_margin,T,h) for T,h in robot_shapes)
                baseline=self.initial_book_pairs.get(name,0.)
                if gap<min(-.002,baseline-book_margin-.002):
                    violations.append('CARRIED_BOOK_ROBOT:'+name)
            lo,hi=book_bounds_in_bin(tip,Tattach,scene)
            if lo[2]<table+.003:violations.append('CARRIED_BOOK_TABLE')
            if lo[0]<ox and hi[0]>-ox and lo[1]<oy and hi[1]>-oy and lo[2]<floor-.006:
                violations.append('BOOK_EXCESSIVE_FLOOR_PENETRATION')
        return {'passed':not violations,'violations':list(dict.fromkeys(violations)),
                'minimum_environment_envelope_gap_m':minimum if np.isfinite(minimum) else 0.,
                'screen_type':'URDF_collision_mesh_OBBs+bin_shell+table+held_or_deposited_box',
                'not_a_continuous_mesh_collision_certificate':True}


def _placement_candidate(model,envelopes,joints,arm,R,Tattach,scene,settings,bias):
    names=[f'arm_{arm}_{i}_joint' for i in range(1,8)]
    chain=model.chain('base_footprint',f'gripper_{arm}_grasping_link')
    q0=np.array([joints[n] for n in names]);T0=model.forward(chain,joints).tip_transform
    if np.linalg.norm(rotation_vector(R@T0[:3,:3].T))>.20:
        raise GeometryError('CARRY_ORIENTATION_DIFFERS_FROM_VALIDATED_GRASP')
    envelopes.set_start(joints)
    envelopes.set_book_start(joints,Tattach)
    B=np.array(scene['base_T_bin']);hx,hy=scene['inner_half_lengths_m']
    relative=apply(transform(R)@Tattach,box_corners(-np.array([.08,.01,.125]),np.array([.08,.01,.125])))
    book_offset=R@Tattach[:3,3]
    # Choose a book-centre point inside the measured opening closest to current
    # position, with whole-book extents and uncertainty included.
    local_vectors=relative@B[:3,:3]
    ext=np.max(np.abs(local_vectors-(book_offset@B[:3,:3])),axis=0)
    allowed=np.array([hx,hy])-ext[:2]-settings['book_margin_m']
    if min(allowed)<0:raise GeometryError('BOOK_ORIENTATION_DOES_NOT_FIT_OPENING')
    current_book=T0[:3,3]+T0[:3,:3]@Tattach[:3,3]
    current_local=apply(np.linalg.inv(B),current_book)[0]
    nearest=np.clip(current_local[:2],-allowed,allowed)
    # Prefer the interior centre, not a near-wall position chosen only for arm
    # reach. Candidate search never relaxes book/robot collision constraints.
    target_local=np.r_[nearest*float(bias),0.]
    book_xy=apply(B,target_local)[0][:2]
    tip_xy=book_xy-book_offset[:2]
    pre_z=scene['rim_z_m']+settings['rim_clearance_m']-relative[:,2].min()
    low_z=scene['floor_z_m']-.002-relative[:,2].min()
    if not .04<pre_z-low_z<.38:raise GeometryError('LOWERING_RANGE_INVALID')
    clear=np.array([T0[0,3],T0[1,3],max(pre_z,T0[2,3])])
    pre=np.r_[tip_xy,pre_z];low=np.r_[tip_xy,low_z]
    retract_up=np.r_[tip_xy,max(pre_z,low_z+.27)]
    away=pre.copy();away[2]=retract_up[2];away[:2]-=settings['retract_away_m']*B[:2,1]*np.sign((B[:2,1]@tip_xy) or 1)
    # Use robot-facing negative base X for the empty-gripper withdrawal; first
    # complete the vertical clearance leg above both bin and deposited book.
    away=retract_up.copy();away[0]-=settings['retract_away_m']
    path_specs=[('pre_place',[clear,pre],True,settings['preplace_speed_mps']),
                ('lower_place',[low],True,settings['lower_speed_mps']),
                ('retract',[retract_up,away],False,settings['retract_speed_mps'])]
    q=q0.copy();position=T0[:3,3].copy();segments={};deposited=None
    for stage,targets,carried,speed in path_specs:
        records=[];previous=q.copy();fixed=dict(joints)
        fixed[f'gripper_{arm}_finger_joint']=(.01 if carried else .04)
        start_q=q.copy();start_position=position.copy()
        worst_delta=0.;max_pos=0.;max_rot=0.;min_gap=float('inf');time_total=0.
        for end in targets:
            dist=float(np.linalg.norm(end-position));steps=max(1,int(math.ceil(dist/settings['cart_step_m'])))
            origin_pos=position.copy()
            for index in range(1,steps+1):
                desired=origin_pos+(end-origin_pos)*index/steps
                solution=solve_ik(model,chain,names,fixed,desired,R,q,
                                  position_tolerance_m=.0025,orientation_tolerance_rad=.025,
                                  max_iterations=260,joint_limit_margin_rad=.025)
                if not solution.success:
                    lower,upper=model.limits(names,margin=.025)
                    for seed in deterministic_seeds(q,lower,upper)[1:]:
                        trial=solve_ik(model,chain,names,fixed,desired,R,seed,
                                       position_tolerance_m=.0025,orientation_tolerance_rad=.025,max_iterations=350)
                        if trial.success and max(abs(np.array(trial.positions)-q))<.24:
                            solution=trial;break
                if not solution.success:raise GeometryError(f'IK_FAIL:{stage}:{index}:{solution.reason}')
                nxt=np.array(solution.positions);delta=float(max(abs(nxt-q)))
                if delta>.24:raise GeometryError('IK_BRANCH_JUMP:'+stage)
                worst_delta=max(worst_delta,delta);max_pos=max(max_pos,solution.position_error_m)
                max_rot=max(max_rot,solution.orientation_error_rad)
                if stage=='retract' and deposited is None:
                    fq=dict(fixed);fq.update(zip(names,q));deposited=model.forward(chain,fq).tip_transform@Tattach
                for sample in interpolate_joint_path(q,nxt,samples=3):
                    fq=dict(fixed);fq.update(zip(names,sample))
                    report=envelopes.check(fq,scene,Tattach,carried,deposited,settings['robot_margin_m'],settings['book_margin_m'])
                    if not report['passed']:
                        raise GeometryError(stage+':'+','.join(report['violations'][:5]))
                    min_gap=min(min_gap,report['minimum_environment_envelope_gap_m'])
                duration=max(.20,float(np.linalg.norm(desired-position))/speed,delta/settings['max_joint_speed_radps'])
                time_total+=duration
                records.append({'positions':nxt.tolist(),'dt_sec':duration,'time_sec':time_total,
                                'target_xyz_m':desired.tolist(),'ik_position_error_m':solution.position_error_m,
                                'ik_orientation_error_rad':solution.orientation_error_rad})
                q=nxt;position=desired
        segments[stage]={'passed':True,'waypoints':records,'start_positions_rad':start_q.tolist(),
                         'start_xyz_m':start_position.tolist(),'target_xyz_m':position.tolist(),
                         'duration_sec':time_total,'max_segment_joint_delta_rad':worst_delta,
                         'max_total_joint_delta_rad':float(max(abs(q-start_q))),
                         'ik_position_error_m':max_pos,'ik_orientation_error_rad':max_rot,
                         'collision_screen':{'passed':True,'minimum_gap_m':min_gap,
                           'sampled_mesh_bounds':True,'carried_book_screened':carried,
                           'deposited_book_screened':not carried}}
    return {'passed':True,'selected_arm':arm,'joint_names':names,
            'target_rotation':R.tolist(),'tip_T_book':Tattach.tolist(),
            'pre_place_xyz_m':pre.tolist(),'lower_place_xyz_m':low.tolist(),
            'retract_xyz_m':away.tolist(),'segments':segments,
            'bin_scene':scene,'initial_joint_positions':dict(joints),
            'collision_meshes_loaded':envelopes.mesh_count,
            'baseline_housing_envelope_overlaps':len(envelopes.initial_pairs),
            'book_dimensions_m':[.16,.02,.25],
            'planned_controlled_lowering_m':float(pre_z-low_z),
            'book_uncertainty_margin_m':settings['book_margin_m'],
            'placement_bias_toward_reachable_edge':bias}


def build_placement_plan(model,envelopes,joints,arm,R,Tattach,scene,settings):
    failures=[]
    for bias in (0.0,0.50,0.75):
        try:
            plan=_placement_candidate(model,envelopes,joints,arm,R,Tattach,scene,settings,bias)
            plan['rejected_placement_candidates']=failures
            plan['grasp_orientation_preserved']=True
            return plan
        except GeometryError as exc:
            failures.append({'bias':bias,'reason':str(exc)})
    raise GeometryError('NO_SAFE_REACHABLE_PLACEMENT_WITH_PRESERVED_ORIENTATION: '+str(failures))


def plan_actual_retract(model,envelopes,joints,arm,R,Tattach,scene,deposited,settings,nominal_plan):
    """Fresh empty-gripper IK from the measured contact-stop/open joint state."""
    names=[f'arm_{arm}_{i}_joint' for i in range(1,8)]
    chain=model.chain('base_footprint',f'gripper_{arm}_grasping_link')
    q=np.array([joints[n] for n in names]);start=q.copy()
    position=model.forward(chain,joints).tip_transform[:3,3];start_position=position.copy()
    target=np.array(nominal_plan['retract_xyz_m'])
    up=position.copy();up[2]=max(position[2]+.27,target[2])
    away=up.copy();away[0]-=settings['retract_away_m']
    records=[];total=0.;count=0;worst=0.;poserr=0.;roterr=0.;minimum=float('inf')
    for endpoint in (up,away):
        origin_pos=position.copy();steps=max(1,int(np.ceil(np.linalg.norm(endpoint-position)/settings['cart_step_m'])))
        for i in range(1,steps+1):
            desired=origin_pos+(endpoint-origin_pos)*i/steps
            sol=solve_ik(model,chain,names,joints,desired,R,q,position_tolerance_m=.0025,
                         orientation_tolerance_rad=.025,max_iterations=300,joint_limit_margin_rad=.025)
            if not sol.success:
                lower,upper=model.limits(names,margin=.025)
                for seed in deterministic_seeds(q,lower,upper)[1:]:
                    test=solve_ik(model,chain,names,joints,desired,R,seed,position_tolerance_m=.0025,
                                  orientation_tolerance_rad=.025,max_iterations=350)
                    if test.success and max(abs(np.array(test.positions)-q))<.24:
                        sol=test;break
            if not sol.success:raise GeometryError('ACTUAL_RETRACT_IK_FAILED:'+sol.reason)
            nxt=np.array(sol.positions);delta=float(max(abs(nxt-q)))
            if delta>.24:raise GeometryError('ACTUAL_RETRACT_IK_BRANCH_JUMP')
            for sample in interpolate_joint_path(q,nxt,8):
                jq=dict(joints);jq.update(zip(names,sample))
                report=envelopes.check(jq,scene,Tattach,False,deposited,settings['robot_margin_m'])
                if not report['passed']:raise GeometryError('ACTUAL_RETRACT_UNSAFE:'+','.join(report['violations'][:5]))
                minimum=min(minimum,report['minimum_environment_envelope_gap_m']);count+=1
            dt=max(.20,float(np.linalg.norm(desired-position))/settings['retract_speed_mps'],delta/settings['max_joint_speed_radps'])
            total+=dt;poserr=max(poserr,sol.position_error_m);roterr=max(roterr,sol.orientation_error_rad);worst=max(worst,delta)
            records.append(dict(positions=nxt.tolist(),time_sec=total,dt_sec=dt,target_xyz_m=desired.tolist(),
                                ik_position_error_m=sol.position_error_m,ik_orientation_error_rad=sol.orientation_error_rad))
            q=nxt;position=desired
    segment=dict(passed=True,waypoints=records,start_positions_rad=start.tolist(),start_xyz_m=start_position.tolist(),
        target_xyz_m=position.tolist(),duration_sec=total,max_segment_joint_delta_rad=worst,
        max_total_joint_delta_rad=float(max(abs(q-start))),ik_position_error_m=poserr,ik_orientation_error_rad=roterr,
        collision_screen=dict(passed=True,minimum_gap_m=minimum,sampled_mesh_bounds=True,carried_book_screened=False,deposited_book_screened=True))
    return dict(passed=True,samples=count,minimum_gap_m=minimum,actual_start_positions_rad=start.tolist(),
                actual_deposited_book_checked=True,new_ik_from_actual_joints=True,segment=segment)


# DAY8_CLEAR_VIEW_TRANSPORT_READY_V1

def _obb_min_z(T,h):
    h=np.asarray(h,dtype=float)
    return float(
        np.min(
            apply(
                T,
                box_corners(-h,h)
            )[:,2]
        )
    )


def _obb_planar_range(T,h,point_xy):
    h=np.asarray(h,dtype=float)
    corners=apply(T,box_corners(-h,h))
    return float(
        np.min(
            np.linalg.norm(
                corners[:,:2]-np.asarray(point_xy,dtype=float),
                axis=1
            )
        )
    )


def _clear_view_screen(
    envelopes,
    joints,
    Tattach,
    bin_xy,
    robot_margin,
    book_margin,
    start_moving_min_z,
    start_book_min_z,
    start_arm_bin_range,
    start_book_bin_range,
):
    shapes=envelopes.poses(joints)
    violations=[]
    minimum=float('inf')

    moving_min_z=float('inf')
    arm_bin_range=float('inf')

    for a in envelopes.moving:
        for T,h in shapes[a]:
            inflated=np.asarray(h)+float(robot_margin)
            moving_min_z=min(
                moving_min_z,
                _obb_min_z(T,inflated))
            arm_bin_range=min(
                arm_bin_range,
                _obb_planar_range(T,inflated,bin_xy))

        for b in shapes:
            if a==b or envelopes.adjacent(a,b):
                continue
            if b in envelopes.moving and b<a:
                continue
            gap=min(
                sat_gap(T,h,U,k)
                for T,h in shapes[a]
                for U,k in shapes[b])
            minimum=min(minimum,gap)
            baseline=envelopes.initial_pairs.get((a,b),0.)
            if gap<min(-.002,baseline-.002):
                violations.append(
                    'CLEAR_VIEW_SELF_ENVELOPE:'+a+':'+b)

    if moving_min_z<start_moving_min_z-.012:
        violations.append('CLEAR_VIEW_ARM_MOVED_LOWER_THAN_SAFE_START')

    if arm_bin_range<start_arm_bin_range-.015:
        violations.append('CLEAR_VIEW_ARM_MOVED_TOWARD_BIN')

    tip=envelopes.model.forward(
        envelopes.model.chain(
            'base_footprint',
            f'gripper_{envelopes.arm}_grasping_link'),
        joints).tip_transform

    Tb=tip@Tattach
    book_h=np.array([.08,.01,.125])+float(book_margin)
    book_min_z=_obb_min_z(Tb,book_h)
    book_bin_range=_obb_planar_range(Tb,book_h,bin_xy)

    if book_min_z<start_book_min_z-.010:
        violations.append('CLEAR_VIEW_BOOK_MOVED_LOWER_THAN_SAFE_START')

    if book_bin_range<start_book_bin_range-.010:
        violations.append('CLEAR_VIEW_BOOK_MOVED_TOWARD_BIN')

    for name,robot_shapes in shapes.items():
        if name.startswith(f'gripper_{envelopes.arm}_'):
            continue
        gap=min(
            sat_gap(Tb,book_h,T,h)
            for T,h in robot_shapes)
        minimum=min(minimum,gap)
        baseline=envelopes.initial_book_pairs.get(name,0.)
        if gap<min(-.002,baseline-float(book_margin)-.002):
            violations.append(
                'CLEAR_VIEW_CARRIED_BOOK_ROBOT:'+name)

    return {
        'passed':not violations,
        'violations':list(dict.fromkeys(violations)),
        'minimum_robot_gap_m':(
            minimum if np.isfinite(minimum) else 0.0),
        'moving_min_z_m':moving_min_z,
        'book_min_z_m':book_min_z,
        'arm_bin_range_m':arm_bin_range,
        'book_bin_range_m':book_bin_range,
    }


def build_clear_view_plan(
    model,
    envelopes,
    joints,
    arm,
    R,
    Tattach,
    bin_xy,
    settings,
):
    """Fast screened clear-view move.

    Solve only candidate endpoints, then screen the complete joint-space path.
    This avoids repeated Cartesian micro-step IK while retaining full FK-based
    collision and carried-book checks.
    """
    names = [f'arm_{arm}_{i}_joint' for i in range(1, 8)]
    chain = model.chain(
        'base_footprint',
        f'gripper_{arm}_grasping_link')

    q0 = np.array([joints[n] for n in names], dtype=float)
    T0 = model.forward(chain, joints).tip_transform

    orientation_error = float(
        np.linalg.norm(rotation_vector(R @ T0[:3, :3].T)))
    if orientation_error > .20:
        raise GeometryError(
            'CLEAR_VIEW_GRASP_ORIENTATION_NOT_PRESERVED')

    bin_xy = np.asarray(bin_xy, dtype=float).reshape(2)
    bin_range = float(np.linalg.norm(bin_xy))

    if not np.isfinite(bin_range) or bin_range < .20:
        raise GeometryError('CLEAR_VIEW_INVALID_BIN_ANCHOR')

    ray = bin_xy / bin_range
    side = np.array([-ray[1], ray[0]])

    envelopes.set_start(joints)
    envelopes.set_book_start(joints, Tattach)

    shapes0 = envelopes.poses(joints)

    start_moving_min_z = min(
        _obb_min_z(
            T,
            np.asarray(h) + float(settings['robot_margin_m']))
        for a in envelopes.moving
        for T, h in shapes0[a]
    )

    start_arm_bin_range = min(
        _obb_planar_range(
            T,
            np.asarray(h) + float(settings['robot_margin_m']),
            bin_xy)
        for a in envelopes.moving
        for T, h in shapes0[a]
    )

    Tb0 = T0 @ Tattach
    book_h = (
        np.array([.08, .01, .125])
        + float(settings['book_margin_m'])
    )

    start_book_min_z = _obb_min_z(Tb0, book_h)
    start_book_bin_range = _obb_planar_range(
        Tb0, book_h, bin_xy)

    book0 = Tb0[:3, 3]
    signed_cross = float(book0[:2] @ side)
    cross_before = abs(signed_cross)

    minimum_cross = float(
        settings['clear_view_min_cross_track_m'])

    # Retry-safe behavior: once the retained book is already outside the
    # required camera/bin sightline, do not push it farther sideways.
    if cross_before >= minimum_cross:
        return {
            'passed': True,
            'selected_arm': arm,
            'joint_names': names,
            'target_rotation': R.tolist(),
            'tip_T_book': Tattach.tolist(),
            'live_bin_anchor_base_xy_m': bin_xy.tolist(),
            'book_cross_track_before_m': cross_before,
            'book_cross_track_after_m': cross_before,
            'required_cross_track_m': minimum_cross,
            'book_bin_range_before_m': float(
                np.linalg.norm(book0[:2] - bin_xy)),
            'book_bin_range_after_m': float(
                np.linalg.norm(book0[:2] - bin_xy)),
            'chosen_lateral_sign': 0.0,
            'chosen_scale': 0.0,
            'segment': {
                'passed': True,
                'waypoints': [],
                'start_positions_rad': q0.tolist(),
                'start_xyz_m': T0[:3,3].tolist(),
                'target_xyz_m': T0[:3,3].tolist(),
                'duration_sec': 0.0,
                'max_segment_joint_delta_rad': 0.0,
                'max_total_joint_delta_rad': 0.0,
                'ik_position_error_m': 0.0,
                'ik_orientation_error_rad': orientation_error,
                'collision_screen': {
                    'passed': True,
                    'samples': 0,
                    'carried_book_screened': True,
                    'complete_joint_path_screened': True,
                    'no_world_pose_used': True,
                    'reason': 'already_clear',
                },
            },
            'rejected_candidates': [],
            'transport_state_machine_independent': True,
            'grasp_orientation_preserved': True,
            'base_motion_required': False,
            'planner_mode':
                'already_clear_no_motion',
        }

    required_cross = max(
        minimum_cross,
        min(cross_before + .05, .22)
    )

    if abs(signed_cross) >= .02:
        preferred = 1.0 if signed_cross >= 0 else -1.0
    else:
        preferred = -1.0 if arm == 'right' else 1.0

    candidates = (
        (1.00, preferred),
        (1.30, preferred),
        (1.00, -preferred),
        (1.30, -preferred),
    )

    lower, upper = model.limits(names, margin=.025)
    base_seeds = deterministic_seeds(q0, lower, upper)[:3]

    accepted = []
    failures = []

    fixed = dict(joints)
    fixed[f'gripper_{arm}_finger_joint'] = float(
        settings['gripper_hold_position_m'])

    for scale, sign in candidates:
        lateral = (
            float(settings['clear_view_lateral_m']) * scale
        )
        back = float(settings['clear_view_back_m'])
        up = float(settings['clear_view_up_m'])

        endpoint = T0[:3, 3].copy()
        endpoint[:2] += -back * ray + sign * lateral * side
        endpoint[2] += up

        solution = None

        for seed_index, seed in enumerate(base_seeds):
            trial = solve_ik(
                model,
                chain,
                names,
                fixed,
                endpoint,
                R,
                seed,
                position_tolerance_m=.003,
                orientation_tolerance_rad=.030,
                max_iterations=140,
                joint_limit_margin_rad=.025,
            )

            if not trial.success:
                continue

            q1 = np.asarray(trial.positions, dtype=float)

            if float(np.max(np.abs(q1 - q0))) > float(
                settings['clear_view_max_joint_delta_rad']
            ):
                continue

            solution = trial
            break

        if solution is None:
            failures.append({
                'scale': scale,
                'sign': sign,
                'reason': 'ENDPOINT_IK_FAILED',
            })
            continue

        q1 = np.asarray(solution.positions, dtype=float)

        minimum_gap = float('inf')
        unsafe = None

        # Screen the COMPLETE commanded joint trajectory.
        for sample in interpolate_joint_path(q0, q1, samples=12):
            state = dict(fixed)
            state.update(zip(names, sample))

            report = _clear_view_screen(
                envelopes,
                state,
                Tattach,
                bin_xy,
                float(settings['robot_margin_m']),
                float(settings['book_margin_m']),
                start_moving_min_z,
                start_book_min_z,
                start_arm_bin_range,
                start_book_bin_range,
            )

            if not report['passed']:
                unsafe = ','.join(report['violations'][:5])
                break

            minimum_gap = min(
                minimum_gap,
                report['minimum_robot_gap_m'])

        if unsafe is not None:
            failures.append({
                'scale': scale,
                'sign': sign,
                'reason': unsafe,
            })
            continue

        final = dict(fixed)
        final.update(zip(names, q1))
        Tf = model.forward(chain, final).tip_transform
        bookf = (Tf @ Tattach)[:3, 3]

        cross_after = abs(float(bookf[:2] @ side))

        range_before = float(
            np.linalg.norm(book0[:2] - bin_xy))
        range_after = float(
            np.linalg.norm(bookf[:2] - bin_xy))

        if cross_after < required_cross:
            failures.append({
                'scale': scale,
                'sign': sign,
                'reason': 'INSUFFICIENT_CROSS_TRACK',
                'cross_after_m': cross_after,
            })
            continue

        if range_after < range_before - .010:
            failures.append({
                'scale': scale,
                'sign': sign,
                'reason': 'BOOK_MOVED_TOWARD_BIN',
            })
            continue

        delta = float(np.max(np.abs(q1 - q0)))
        distance = float(
            np.linalg.norm(endpoint - T0[:3, 3]))

        duration = max(
            .35,
            distance / float(settings['clear_view_speed_mps']),
            delta / float(settings['max_joint_speed_radps']),
        )

        segment = {
            'passed': True,
            'start_positions_rad': q0.tolist(),
            'start_xyz_m': T0[:3, 3].tolist(),
            'target_xyz_m': endpoint.tolist(),
            'duration_sec': duration,
            'max_segment_joint_delta_rad': delta,
            'max_total_joint_delta_rad': delta,
            'ik_position_error_m':
                float(solution.position_error_m),
            'ik_orientation_error_rad':
                float(solution.orientation_error_rad),
            'waypoints': [{
                'positions': q1.tolist(),
                'dt_sec': duration,
                'time_sec': duration,
                'target_xyz_m': endpoint.tolist(),
                'ik_position_error_m':
                    float(solution.position_error_m),
                'ik_orientation_error_rad':
                    float(solution.orientation_error_rad),
            }],
            'collision_screen': {
                'passed': True,
                'samples': 12,
                'minimum_gap_m': (
                    minimum_gap
                    if np.isfinite(minimum_gap)
                    else 0.0
                ),
                'carried_book_screened': True,
                'complete_joint_path_screened': True,
                'no_world_pose_used': True,
            },
        }

        score = (
            duration
            + .5 * delta
            - .25 * (cross_after - required_cross)
        )

        accepted.append((
            score,
            {
                'passed': True,
                'selected_arm': arm,
                'joint_names': names,
                'target_rotation': R.tolist(),
                'tip_T_book': Tattach.tolist(),
                'live_bin_anchor_base_xy_m':
                    bin_xy.tolist(),
                'book_cross_track_before_m':
                    cross_before,
                'book_cross_track_after_m':
                    cross_after,
                'required_cross_track_m':
                    required_cross,
                'book_bin_range_before_m':
                    range_before,
                'book_bin_range_after_m':
                    range_after,
                'chosen_lateral_sign': sign,
                'chosen_scale': scale,
                'segment': segment,
                'rejected_candidates':
                    failures.copy(),
                'transport_state_machine_independent':
                    True,
                'grasp_orientation_preserved':
                    True,
                'base_motion_required':
                    False,
                'planner_mode':
                    'endpoint_IK_plus_complete_joint_path_screen',
            }
        ))

    if not accepted:
        raise GeometryError(
            'NO_SAFE_FAST_CLEAR_VIEW_POSE: '
            + str(failures))

    return min(accepted, key=lambda item: item[0])[1]


# ============================================================
# FAST DAY8 PLACEMENT
# Endpoint IK + dense joint-path collision screening.
#
# This intentionally does NOT call the old exhaustive
# Cartesian-step planner.
# ============================================================

def _fast_day8_solve(
    model,
    envelopes,
    chain,
    names,
    fixed,
    q,
    current_xyz,
    target_xyz,
    R,
    scene,
    Tattach,
    settings,
    carried,
    deposited_T,
    maximum_jump,
    deadline,
):
    import time

    if time.perf_counter() >= deadline:
        raise GeometryError('FAST_PLAN_BUDGET_EXCEEDED')

    iterations = int(
        settings.get(
            'fast_plan_ik_iterations',
            90,
        )
    )

    solution = solve_ik(
        model,
        chain,
        names,
        fixed,
        target_xyz,
        R,
        q,
        position_tolerance_m=.003,
        orientation_tolerance_rad=.035,
        max_iterations=iterations,
        joint_limit_margin_rad=.025,
    )

    # Bounded retry: at most two alternate deterministic seeds.
    if not solution.success:

        lower, upper = model.limits(
            names,
            margin=.025,
        )

        seeds = deterministic_seeds(
            q,
            lower,
            upper,
        )

        retry_count = max(
            0,
            int(
                settings.get(
                    'fast_stage_seed_retries',
                    2,
                )
            ),
        )

        for seed in seeds[1:1 + retry_count]:

            if time.perf_counter() >= deadline:
                raise GeometryError(
                    'FAST_PLAN_BUDGET_EXCEEDED'
                )

            trial = solve_ik(
                model,
                chain,
                names,
                fixed,
                target_xyz,
                R,
                seed,
                position_tolerance_m=.003,
                orientation_tolerance_rad=.035,
                max_iterations=iterations,
                joint_limit_margin_rad=.025,
            )

            if trial.success:
                solution = trial
                break

    if not solution.success:
        raise GeometryError(
            'FAST_IK_FAIL:'
            + str(solution.reason)
        )

    nxt = np.asarray(
        solution.positions,
        dtype=float,
    )

    delta = float(
        np.max(
            np.abs(
                nxt - q
            )
        )
    )

    if delta > maximum_jump:
        raise GeometryError(
            'FAST_IK_BRANCH_JUMP:'
            + str(delta)
        )

    minimum = float('inf')

    samples = int(
        settings.get(
            'fast_plan_joint_samples',
            8,
        )
    )

    samples = max(
        5,
        min(
            14,
            samples,
        ),
    )

    for sample in interpolate_joint_path(
        q,
        nxt,
        samples=samples,
    ):

        if time.perf_counter() >= deadline:
            raise GeometryError(
                'FAST_PLAN_BUDGET_EXCEEDED'
            )

        fq = dict(fixed)

        fq.update(
            zip(
                names,
                sample,
            )
        )

        report = envelopes.check(
            fq,
            scene,
            Tattach,
            carried,
            deposited_T,
            settings['robot_margin_m'],
            settings['book_margin_m'],
        )

        if not report['passed']:
            raise GeometryError(
                'FAST_COLLISION:'
                + ','.join(
                    report['violations'][:5]
                )
            )

        minimum = min(
            minimum,
            report[
                'minimum_environment_envelope_gap_m'
            ],
        )

    distance = float(
        np.linalg.norm(
            np.asarray(target_xyz)
            - np.asarray(current_xyz)
        )
    )

    speed = float(
        settings[
            'preplace_speed_mps'
            if carried
            else 'retract_speed_mps'
        ]
    )

    duration = max(
        .20,
        distance / max(speed, 1.0e-6),
        delta
        / settings['max_joint_speed_radps'],
    )

    record = {
        'positions':
            nxt.tolist(),

        'dt_sec':
            duration,

        'time_sec':
            duration,

        'target_xyz_m':
            np.asarray(
                target_xyz,
                dtype=float,
            ).tolist(),

        'ik_position_error_m':
            float(
                solution.position_error_m
            ),

        'ik_orientation_error_rad':
            float(
                solution.orientation_error_rad
            ),
    }

    return (
        nxt,
        record,
        minimum,
        float(
            solution.position_error_m
        ),
        float(
            solution.orientation_error_rad
        ),
    )


def build_fast_placement_plan(
    model,
    envelopes,
    joints,
    arm,
    R,
    Tattach,
    scene,
    settings,
):
    """Bounded fast placement from gravity-supported handoff.

    The starting wrist may differ significantly from the original
    Day5 grasp orientation.  The first pre-place trajectory both:
      1. restores the required placement orientation, and
      2. moves immediately toward the bin.

    There is no separate upright waiting state.
    """

    import time

    started = time.perf_counter()

    budget = float(
        settings.get(
            'fast_plan_budget_sec',
            3.0,
        )
    )

    deadline = (
        started
        + budget
    )

    names = [
        f'arm_{arm}_{i}_joint'
        for i in range(1, 8)
    ]

    chain = model.chain(
        'base_footprint',
        f'gripper_{arm}_grasping_link',
    )

    q0 = np.asarray(
        [
            joints[n]
            for n in names
        ],
        dtype=float,
    )

    T0 = model.forward(
        chain,
        joints,
    ).tip_transform

    envelopes.set_start(
        joints
    )

    envelopes.set_book_start(
        joints,
        Tattach,
    )

    B = np.asarray(
        scene['base_T_bin'],
        dtype=float,
    )

    hx, hy = (
        scene[
            'inner_half_lengths_m'
        ]
    )

    # Target book extents use the ORIGINAL validated Day5
    # placement orientation R.
    relative = apply(
        transform(R) @ Tattach,
        box_corners(
            -np.asarray(
                [.08, .01, .125]
            ),
            np.asarray(
                [.08, .01, .125]
            ),
        ),
    )

    book_offset = (
        R @ Tattach[:3, 3]
    )

    local_vectors = (
        relative @ B[:3, :3]
    )

    ext = np.max(
        np.abs(
            local_vectors
            - (
                book_offset
                @ B[:3, :3]
            )
        ),
        axis=0,
    )

    allowed = (
        np.asarray(
            [hx, hy]
        )
        - ext[:2]
        - settings[
            'book_margin_m'
        ]
    )

    if np.min(allowed) < 0:
        raise GeometryError(
            'FAST_BOOK_ORIENTATION_DOES_NOT_FIT_OPENING'
        )

    current_book = (
        T0[:3, 3]
        + T0[:3, :3]
        @ Tattach[:3, 3]
    )

    current_local = apply(
        np.linalg.inv(B),
        current_book,
    )[0]

    nearest = np.clip(
        current_local[:2],
        -allowed,
        allowed,
    )

    failures = []

    # Usually centre passes immediately.
    # Two bounded alternatives exist, but there is NO exhaustive search.
    for bias in (
        0.0,
        0.50,
        0.75,
    ):

        if time.perf_counter() >= deadline:
            break

        try:
            target_local = np.r_[
                nearest * float(bias),
                0.0,
            ]

            book_xy = apply(
                B,
                target_local,
            )[0][:2]

            tip_xy = (
                book_xy
                - book_offset[:2]
            )

            pre_z = (
                scene['rim_z_m']
                + settings[
                    'rim_clearance_m'
                ]
                - relative[:, 2].min()
            )

            low_z = (
                scene['floor_z_m']
                - .002
                - relative[:, 2].min()
            )

            if not (
                .04
                < pre_z - low_z
                < .38
            ):
                raise GeometryError(
                    'FAST_LOWERING_RANGE_INVALID'
                )

            pre = np.r_[
                tip_xy,
                pre_z,
            ]

            low = np.r_[
                tip_xy,
                low_z,
            ]

            retract_up = np.r_[
                tip_xy,
                max(
                    pre_z,
                    low_z + .27,
                ),
            ]

            away = retract_up.copy()

            away[0] -= settings[
                'retract_away_m'
            ]

            fixed = dict(
                joints
            )

            fixed[
                f'gripper_{arm}_finger_joint'
            ] = float(
                joints.get(
                    f'gripper_{arm}_finger_joint',
                    0.0,
                )
            )

            segments = {}

            # ------------------------------------------------
            # PREPLACE
            #
            # One screened trajectory:
            # gravity-supported pose -> original placement
            # orientation + above-bin preplace.
            # ------------------------------------------------

            q = q0.copy()

            position = (
                T0[:3, 3].copy()
            )

            q_pre, rec, gap, pe, re = (
                _fast_day8_solve(
                    model,
                    envelopes,
                    chain,
                    names,
                    fixed,
                    q,
                    position,
                    pre,
                    R,
                    scene,
                    Tattach,
                    settings,
                    True,
                    None,
                    1.80,
                    deadline,
                )
            )

            segments['pre_place'] = {
                'passed':
                    True,

                'waypoints':
                    [rec],

                'start_positions_rad':
                    q.tolist(),

                'start_xyz_m':
                    position.tolist(),

                'target_xyz_m':
                    pre.tolist(),

                'duration_sec':
                    rec['dt_sec'],

                'max_segment_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_pre - q
                            )
                        )
                    ),

                'max_total_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_pre - q
                            )
                        )
                    ),

                'ik_position_error_m':
                    pe,

                'ik_orientation_error_rad':
                    re,

                'collision_screen': {
                    'passed':
                        True,
                    'minimum_gap_m':
                        gap,
                    'fast_screened_joint_path':
                        True,
                    'carried_book_screened':
                        True,
                },
            }

            # ------------------------------------------------
            # LOWER
            # ------------------------------------------------

            q_low, rec_low, gap, pe, re = (
                _fast_day8_solve(
                    model,
                    envelopes,
                    chain,
                    names,
                    fixed,
                    q_pre,
                    pre,
                    low,
                    R,
                    scene,
                    Tattach,
                    settings,
                    True,
                    None,
                    .65,
                    deadline,
                )
            )

            # Lowering uses its dedicated conservative speed.
            lower_distance = float(
                np.linalg.norm(
                    low - pre
                )
            )

            lower_duration = max(
                .20,
                lower_distance
                / settings[
                    'lower_speed_mps'
                ],
                float(
                    np.max(
                        np.abs(
                            q_low - q_pre
                        )
                    )
                )
                / settings[
                    'max_joint_speed_radps'
                ],
            )

            rec_low['dt_sec'] = (
                lower_duration
            )

            rec_low['time_sec'] = (
                lower_duration
            )

            segments['lower_place'] = {
                'passed':
                    True,

                'waypoints':
                    [rec_low],

                'start_positions_rad':
                    q_pre.tolist(),

                'start_xyz_m':
                    pre.tolist(),

                'target_xyz_m':
                    low.tolist(),

                'duration_sec':
                    lower_duration,

                'max_segment_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_low - q_pre
                            )
                        )
                    ),

                'max_total_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_low - q_pre
                            )
                        )
                    ),

                'ik_position_error_m':
                    pe,

                'ik_orientation_error_rad':
                    re,

                'collision_screen': {
                    'passed':
                        True,
                    'minimum_gap_m':
                        gap,
                    'fast_screened_joint_path':
                        True,
                    'carried_book_screened':
                        True,
                },
            }

            # Deposited book pose used for empty-hand retreat screening.
            fq_low = dict(
                fixed
            )

            fq_low.update(
                zip(
                    names,
                    q_low,
                )
            )

            deposited = (
                model.forward(
                    chain,
                    fq_low,
                ).tip_transform
                @ Tattach
            )

            # ------------------------------------------------
            # RETRACT
            #
            # Only a fast nominal retract is prepared here.
            # Day8 still performs its existing fresh actual-state
            # re-screen after release.
            # ------------------------------------------------

            open_fixed = dict(
                fixed
            )

            open_fixed[
                f'gripper_{arm}_finger_joint'
            ] = .04

            q_up, rec_up, gap1, pe1, re1 = (
                _fast_day8_solve(
                    model,
                    envelopes,
                    chain,
                    names,
                    open_fixed,
                    q_low,
                    low,
                    retract_up,
                    R,
                    scene,
                    Tattach,
                    settings,
                    False,
                    deposited,
                    .75,
                    deadline,
                )
            )

            q_away, rec_away, gap2, pe2, re2 = (
                _fast_day8_solve(
                    model,
                    envelopes,
                    chain,
                    names,
                    open_fixed,
                    q_up,
                    retract_up,
                    away,
                    R,
                    scene,
                    Tattach,
                    settings,
                    False,
                    deposited,
                    .75,
                    deadline,
                )
            )

            rec_away[
                'time_sec'
            ] = (
                rec_up['dt_sec']
                + rec_away['dt_sec']
            )

            segments['retract'] = {
                'passed':
                    True,

                'waypoints': [
                    rec_up,
                    rec_away,
                ],

                'start_positions_rad':
                    q_low.tolist(),

                'start_xyz_m':
                    low.tolist(),

                'target_xyz_m':
                    away.tolist(),

                'duration_sec':
                    (
                        rec_up['dt_sec']
                        + rec_away['dt_sec']
                    ),

                'max_segment_joint_delta_rad':
                    max(
                        float(
                            np.max(
                                np.abs(
                                    q_up - q_low
                                )
                            )
                        ),
                        float(
                            np.max(
                                np.abs(
                                    q_away - q_up
                                )
                            )
                        ),
                    ),

                'max_total_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_away - q_low
                            )
                        )
                    ),

                'ik_position_error_m':
                    max(
                        pe1,
                        pe2,
                    ),

                'ik_orientation_error_rad':
                    max(
                        re1,
                        re2,
                    ),

                'collision_screen': {
                    'passed':
                        True,
                    'minimum_gap_m':
                        min(
                            gap1,
                            gap2,
                        ),
                    'fast_screened_joint_path':
                        True,
                    'deposited_book_screened':
                        True,
                },
            }

            wall = (
                time.perf_counter()
                - started
            )

            print(
                '[DAY8 FAST PLAN][PASS] '
                f'bias={bias} '
                f'wall={wall:.3f}s',
                flush=True,
            )

            return {
                'passed':
                    True,

                'selected_arm':
                    arm,

                'joint_names':
                    names,

                'target_rotation':
                    R.tolist(),

                'tip_T_book':
                    Tattach.tolist(),

                'pre_place_xyz_m':
                    pre.tolist(),

                'lower_place_xyz_m':
                    low.tolist(),

                'retract_xyz_m':
                    away.tolist(),

                'segments':
                    segments,

                'bin_scene':
                    scene,

                'initial_joint_positions':
                    dict(joints),

                'collision_meshes_loaded':
                    envelopes.mesh_count,

                'baseline_housing_envelope_overlaps':
                    len(
                        envelopes.initial_pairs
                    ),

                'book_dimensions_m':
                    [.16, .02, .25],

                'planned_controlled_lowering_m':
                    float(
                        pre_z - low_z
                    ),

                'book_uncertainty_margin_m':
                    settings[
                        'book_margin_m'
                    ],

                'placement_bias_toward_reachable_edge':
                    bias,

                'fast_planner':
                    True,

                'planning_wall_sec':
                    wall,

                'supported_orientation_at_planning_start':
                    True,
            }

        except GeometryError as exc:

            failures.append({
                'bias':
                    bias,

                'reason':
                    str(exc),
            })

            print(
                '[DAY8 FAST PLAN][REJECT] '
                f'bias={bias} '
                f'reason={exc}',
                flush=True,
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    raise GeometryError(
        'FAST_PLACEMENT_FAILED_WITHIN_BUDGET: '
        f'wall={elapsed:.3f}s '
        f'failures={failures}'
    )


# ============================================================
# DAY8 FAST STAGED PLACEMENT V1
#
# Execute path:
#   plan PREPLACE only
#   physically move
#   plan LOWER from actual settled joints
#   physically lower
#   release
#   existing actual-state retract replanner
# ============================================================

def _fast_stage_geometry(
    model,
    joints,
    arm,
    R,
    Tattach,
    scene,
    settings,
    bias,
):
    names = [
        f'arm_{arm}_{i}_joint'
        for i in range(1, 8)
    ]

    chain = model.chain(
        'base_footprint',
        f'gripper_{arm}_grasping_link',
    )

    T0 = model.forward(
        chain,
        joints,
    ).tip_transform

    B = np.asarray(
        scene['base_T_bin'],
        dtype=float,
    )

    hx, hy = scene[
        'inner_half_lengths_m'
    ]

    # Final placement uses the validated Day5 book orientation.
    relative = apply(
        transform(R) @ Tattach,
        box_corners(
            -np.asarray(
                [.08, .01, .125],
                dtype=float,
            ),
            np.asarray(
                [.08, .01, .125],
                dtype=float,
            ),
        ),
    )

    book_offset = (
        R @ Tattach[:3, 3]
    )

    local_vectors = (
        relative @ B[:3, :3]
    )

    ext = np.max(
        np.abs(
            local_vectors
            - (
                book_offset
                @ B[:3, :3]
            )
        ),
        axis=0,
    )

    allowed = (
        np.asarray(
            [hx, hy],
            dtype=float,
        )
        - ext[:2]
        - float(
            settings[
                'book_margin_m'
            ]
        )
    )

    if np.min(allowed) < 0:
        raise GeometryError(
            'FAST_BOOK_ORIENTATION_DOES_NOT_FIT_OPENING'
        )

    # Use the actual CURRENT carried-book location only to
    # decide which legal opening position is nearest.
    current_book = (
        T0[:3, 3]
        + T0[:3, :3]
        @ Tattach[:3, 3]
    )

    current_local = apply(
        np.linalg.inv(B),
        current_book,
    )[0]

    nearest = np.clip(
        current_local[:2],
        -allowed,
        allowed,
    )

    target_local = np.r_[
        nearest * float(bias),
        0.0,
    ]

    book_xy = apply(
        B,
        target_local,
    )[0][:2]

    tip_xy = (
        book_xy
        - book_offset[:2]
    )

    pre_z = (
        scene['rim_z_m']
        + settings[
            'rim_clearance_m'
        ]
        - relative[:, 2].min()
    )

    low_z = (
        scene['floor_z_m']
        - .002
        - relative[:, 2].min()
    )

    lowering = (
        pre_z - low_z
    )

    if not .04 < lowering < .38:
        raise GeometryError(
            'FAST_LOWERING_RANGE_INVALID'
        )

    pre = np.r_[
        tip_xy,
        pre_z,
    ]

    low = np.r_[
        tip_xy,
        low_z,
    ]

    retract_up = np.r_[
        tip_xy,
        max(
            pre_z,
            low_z + .27,
        ),
    ]

    away = retract_up.copy()
    away[0] -= settings[
        'retract_away_m'
    ]

    return {
        'names':
            names,

        'chain':
            chain,

        'T0':
            T0,

        'pre':
            pre,

        'low':
            low,

        'retract':
            away,

        'lowering_m':
            float(lowering),

        'bias':
            float(bias),
    }


def build_fast_preplace_plan(
    model,
    envelopes,
    joints,
    arm,
    R,
    Tattach,
    scene,
    settings,
):
    """Plan ONLY the next physical motion: supported carry -> pre-place."""

    import time

    started = time.perf_counter()

    budget = float(
        settings.get(
            'fast_preplace_budget_sec',
            2.5,
        )
    )

    deadline = (
        started + budget
    )

    failures = []

    # Central placement first. Alternatives are attempted only if
    # enough budget remains.
    for bias in (
        0.0,
        0.50,
        0.75,
    ):

        if time.perf_counter() >= deadline:
            break

        try:
            g = _fast_stage_geometry(
                model,
                joints,
                arm,
                R,
                Tattach,
                scene,
                settings,
                bias,
            )

            names = g['names']
            chain = g['chain']
            T0 = g['T0']

            q0 = np.asarray(
                [
                    joints[n]
                    for n in names
                ],
                dtype=float,
            )

            envelopes.set_start(
                joints
            )

            envelopes.set_book_start(
                joints,
                Tattach,
            )

            stage_settings = dict(
                settings
            )

            stage_settings[
                'fast_plan_ik_iterations'
            ] = int(
                settings.get(
                    'fast_stage_ik_iterations',
                    45,
                )
            )

            stage_settings[
                'fast_plan_joint_samples'
            ] = int(
                settings.get(
                    'fast_stage_joint_samples',
                    7,
                )
            )

            stage_settings[
                'fast_stage_seed_retries'
            ] = int(
                settings.get(
                    'fast_stage_seed_retries',
                    0,
                )
            )

            q_pre, record, gap, pe, re = (
                _fast_day8_solve(
                    model,
                    envelopes,
                    chain,
                    names,
                    dict(joints),
                    q0,
                    T0[:3, 3],
                    g['pre'],
                    R,
                    scene,
                    Tattach,
                    stage_settings,
                    True,
                    None,
                    1.80,
                    deadline,
                )
            )

            duration = max(
                .20,
                float(
                    np.linalg.norm(
                        g['pre']
                        - T0[:3, 3]
                    )
                )
                / settings[
                    'preplace_speed_mps'
                ],
                float(
                    np.max(
                        np.abs(
                            q_pre - q0
                        )
                    )
                )
                / settings[
                    'max_joint_speed_radps'
                ],
            )

            record[
                'dt_sec'
            ] = duration

            record[
                'time_sec'
            ] = duration

            segment = {
                'passed':
                    True,

                'waypoints':
                    [record],

                'start_positions_rad':
                    q0.tolist(),

                'start_xyz_m':
                    T0[:3, 3].tolist(),

                'target_xyz_m':
                    g['pre'].tolist(),

                'duration_sec':
                    duration,

                'max_segment_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_pre - q0
                            )
                        )
                    ),

                'max_total_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_pre - q0
                            )
                        )
                    ),

                'ik_position_error_m':
                    pe,

                'ik_orientation_error_rad':
                    re,

                'collision_screen': {
                    'passed':
                        True,

                    'minimum_gap_m':
                        gap,

                    'fast_screened_joint_path':
                        True,

                    'carried_book_screened':
                        True,
                },
            }

            elapsed = (
                time.perf_counter()
                - started
            )

            print(
                '[DAY8 FAST PREPLACE][PASS] '
                f'bias={bias:.2f} '
                f'wall={elapsed:.3f}s '
                f'joint_delta='
                f'{segment["max_total_joint_delta_rad"]:.3f}',
                flush=True,
            )

            return {
                'passed':
                    True,

                'selected_arm':
                    arm,

                'joint_names':
                    names,

                'target_rotation':
                    R.tolist(),

                'tip_T_book':
                    Tattach.tolist(),

                'pre_place_xyz_m':
                    g['pre'].tolist(),

                'lower_place_xyz_m':
                    g['low'].tolist(),

                # Required later by the existing actual-state
                # empty-hand retract replanner.
                'retract_xyz_m':
                    g['retract'].tolist(),

                'segments': {
                    'pre_place':
                        segment,
                },

                'bin_scene':
                    scene,

                'initial_joint_positions':
                    dict(joints),

                'collision_meshes_loaded':
                    envelopes.mesh_count,

                'baseline_housing_envelope_overlaps':
                    len(
                        envelopes.initial_pairs
                    ),

                'book_dimensions_m':
                    [.16, .02, .25],

                'planned_controlled_lowering_m':
                    g['lowering_m'],

                'book_uncertainty_margin_m':
                    settings[
                        'book_margin_m'
                    ],

                'placement_bias_toward_reachable_edge':
                    bias,

                'fast_staged_planning':
                    True,

                'preplace_planning_wall_sec':
                    elapsed,
            }

        except GeometryError as exc:

            failures.append({
                'bias':
                    bias,

                'reason':
                    str(exc),
            })

            print(
                '[DAY8 FAST PREPLACE][REJECT] '
                f'bias={bias:.2f} '
                f'reason={exc}',
                flush=True,
            )

            if (
                'BUDGET_EXCEEDED'
                in str(exc)
            ):
                break

    elapsed = (
        time.perf_counter()
        - started
    )

    raise GeometryError(
        'FAST_PREPLACE_FAILED: '
        f'wall={elapsed:.3f}s '
        f'failures={failures}'
    )


def build_fast_lower_plan(
    model,
    envelopes,
    joints,
    arm,
    R,
    Tattach,
    scene,
    settings,
    nominal_plan,
):
    """Plan ONLY lowering from the ACTUAL settled pre-place joints."""

    import time

    started = time.perf_counter()

    budget = float(
        settings.get(
            'fast_lower_budget_sec',
            1.5,
        )
    )

    deadline = (
        started + budget
    )

    names = [
        f'arm_{arm}_{i}_joint'
        for i in range(1, 8)
    ]

    chain = model.chain(
        'base_footprint',
        f'gripper_{arm}_grasping_link',
    )

    q0 = np.asarray(
        [
            joints[n]
            for n in names
        ],
        dtype=float,
    )

    T0 = model.forward(
        chain,
        joints,
    ).tip_transform

    target = np.asarray(
        nominal_plan[
            'lower_place_xyz_m'
        ],
        dtype=float,
    )

    envelopes.set_start(
        joints
    )

    envelopes.set_book_start(
        joints,
        Tattach,
    )

    stage_settings = dict(
        settings
    )

    stage_settings[
        'fast_plan_ik_iterations'
    ] = int(
        settings.get(
            'fast_stage_ik_iterations',
            45,
        )
    )

    stage_settings[
        'fast_plan_joint_samples'
    ] = int(
        settings.get(
            'fast_stage_joint_samples',
            7,
        )
    )

    stage_settings[
        'fast_stage_seed_retries'
    ] = 0

    q_low, record, gap, pe, re = (
        _fast_day8_solve(
            model,
            envelopes,
            chain,
            names,
            dict(joints),
            q0,
            T0[:3, 3],
            target,
            R,
            scene,
            Tattach,
            stage_settings,
            True,
            None,
            .70,
            deadline,
        )
    )

    distance = float(
        np.linalg.norm(
            target - T0[:3, 3]
        )
    )

    delta = float(
        np.max(
            np.abs(
                q_low - q0
            )
        )
    )

    duration = max(
        .20,
        distance
        / settings[
            'lower_speed_mps'
        ],
        delta
        / settings[
            'max_joint_speed_radps'
        ],
    )

    record[
        'dt_sec'
    ] = duration

    record[
        'time_sec'
    ] = duration

    segment = {
        'passed':
            True,

        'waypoints':
            [record],

        'start_positions_rad':
            q0.tolist(),

        'start_xyz_m':
            T0[:3, 3].tolist(),

        'target_xyz_m':
            target.tolist(),

        'duration_sec':
            duration,

        'max_segment_joint_delta_rad':
            delta,

        'max_total_joint_delta_rad':
            delta,

        'ik_position_error_m':
            pe,

        'ik_orientation_error_rad':
            re,

        'collision_screen': {
            'passed':
                True,

            'minimum_gap_m':
                gap,

            'fast_screened_joint_path':
                True,

            'carried_book_screened':
                True,
        },
    }

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        '[DAY8 FAST LOWER][PASS] '
        f'wall={elapsed:.3f}s '
        f'distance={distance:.3f}m '
        f'joint_delta={delta:.3f}',
        flush=True,
    )

    return {
        'passed':
            True,

        'segment':
            segment,

        'planning_wall_sec':
            elapsed,

        'new_ik_from_actual_preplace_joints':
            True,
    }


# ============================================================
# DAY8 FAST PREPLACE V2
#
# Do NOT ask IK to rediscover the 90-degree carry restoration.
#
# Waypoint 1:
#   exact compact-carry joints already solved/screened by FAST67
#
# Waypoint 2:
#   fresh bin-relative pre-place IK from that validated pose
#
# Both are sent as ONE trajectory.  There is no upright dwell.
# ============================================================

def build_fast_preplace_plan(
    model,
    envelopes,
    joints,
    arm,
    R,
    Tattach,
    scene,
    settings,
):
    import time

    started = time.perf_counter()

    budget = float(
        settings.get(
            'fast_preplace_budget_sec',
            4.0,
        )
    )

    deadline = started + budget

    names = [
        f'arm_{arm}_{i}_joint'
        for i in range(1, 8)
    ]

    chain = model.chain(
        'base_footprint',
        f'gripper_{arm}_grasping_link',
    )

    q_supported = np.asarray(
        [
            joints[n]
            for n in names
        ],
        dtype=float,
    )

    restore_raw = settings.get(
        'known_restore_positions_rad'
    )

    if (
        not isinstance(
            restore_raw,
            (list, tuple),
        )
        or len(restore_raw) != 7
    ):
        raise GeometryError(
            'KNOWN_COMPACT_CARRY_JOINTS_MISSING'
        )

    q_restore = np.asarray(
        restore_raw,
        dtype=float,
    )

    T_supported = model.forward(
        chain,
        joints,
    ).tip_transform

    # --------------------------------------------------------
    # 1. Re-screen the KNOWN reverse support path in the
    # CURRENT measured Day8 scene.
    # --------------------------------------------------------

    envelopes.set_start(
        joints
    )

    envelopes.set_book_start(
        joints,
        Tattach,
    )

    fixed = dict(joints)

    restore_min_gap = float(
        'inf'
    )

    restore_samples = 9

    for sample in interpolate_joint_path(
        q_supported,
        q_restore,
        samples=restore_samples,
    ):
        if time.perf_counter() >= deadline:
            raise GeometryError(
                'FAST_PREPLACE_BUDGET_EXCEEDED_DURING_RESTORE_SCREEN'
            )

        jq = dict(fixed)

        jq.update(
            zip(
                names,
                sample,
            )
        )

        report = envelopes.check(
            jq,
            scene,
            Tattach,
            True,
            None,
            settings[
                'robot_margin_m'
            ],
            settings[
                'book_margin_m'
            ],
        )

        if not report['passed']:
            raise GeometryError(
                'KNOWN_RESTORE_PATH_UNSAFE:'
                + ','.join(
                    report[
                        'violations'
                    ][:5]
                )
            )

        restore_min_gap = min(
            restore_min_gap,
            report[
                'minimum_environment_envelope_gap_m'
            ],
        )

    restore_joints = dict(
        fixed
    )

    restore_joints.update(
        zip(
            names,
            q_restore,
        )
    )

    T_restore = model.forward(
        chain,
        restore_joints,
    ).tip_transform

    restore_orientation_error = float(
        np.linalg.norm(
            rotation_vector(
                R
                @ T_restore[
                    :3,
                    :3
                ].T
            )
        )
    )

    if restore_orientation_error > .10:
        raise GeometryError(
            'KNOWN_COMPACT_CARRY_ORIENTATION_MISMATCH:'
            + str(
                restore_orientation_error
            )
        )

    restore_delta = float(
        np.max(
            np.abs(
                q_restore
                - q_supported
            )
        )
    )

    restore_tip_distance = float(
        np.linalg.norm(
            T_restore[:3, 3]
            - T_supported[:3, 3]
        )
    )

    restore_duration = max(
        .20,
        restore_tip_distance
        / settings[
            'preplace_speed_mps'
        ],
        restore_delta
        / settings[
            'max_joint_speed_radps'
        ],
    )

    restore_record = {
        'positions':
            q_restore.tolist(),

        'dt_sec':
            restore_duration,

        'time_sec':
            restore_duration,

        'target_xyz_m':
            T_restore[
                :3,
                3
            ].tolist(),

        'ik_position_error_m':
            0.0,

        'ik_orientation_error_rad':
            restore_orientation_error,

        'known_fast67_waypoint':
            True,
    }

    # --------------------------------------------------------
    # 2. Solve only the remaining compact-carry -> pre-place
    # move. Starting orientation now already matches R.
    # --------------------------------------------------------

    failures = []

    for bias in (
        0.0,
        0.50,
        0.75,
    ):

        if time.perf_counter() >= deadline:
            break

        try:
            g = _fast_stage_geometry(
                model,
                joints,
                arm,
                R,
                Tattach,
                scene,
                settings,
                bias,
            )

            stage_settings = dict(
                settings
            )

            # More realistic than the previous 45-iteration
            # zero-retry solve, while still strictly bounded.
            stage_settings[
                'fast_plan_ik_iterations'
            ] = 110

            stage_settings[
                'fast_plan_joint_samples'
            ] = 8

            stage_settings[
                'fast_stage_seed_retries'
            ] = 1

            (
                q_pre,
                pre_record,
                pre_gap,
                position_error,
                orientation_error,
            ) = _fast_day8_solve(
                model,
                envelopes,
                chain,
                names,
                fixed,
                q_restore,
                T_restore[:3, 3],
                g['pre'],
                R,
                scene,
                Tattach,
                stage_settings,
                True,
                None,
                1.40,
                deadline,
            )

            pre_delta = float(
                np.max(
                    np.abs(
                        q_pre
                        - q_restore
                    )
                )
            )

            pre_distance = float(
                np.linalg.norm(
                    g['pre']
                    - T_restore[
                        :3,
                        3
                    ]
                )
            )

            pre_duration = max(
                .20,
                pre_distance
                / settings[
                    'preplace_speed_mps'
                ],
                pre_delta
                / settings[
                    'max_joint_speed_radps'
                ],
            )

            pre_record[
                'dt_sec'
            ] = pre_duration

            # IMPORTANT:
            # cumulative controller time.
            # Robot passes through q_restore and continues
            # directly to q_pre without waiting there.
            pre_record[
                'time_sec'
            ] = (
                restore_duration
                + pre_duration
            )

            segment = {
                'passed':
                    True,

                'waypoints': [
                    restore_record,
                    pre_record,
                ],

                'start_positions_rad':
                    q_supported.tolist(),

                'start_xyz_m':
                    T_supported[
                        :3,
                        3
                    ].tolist(),

                'target_xyz_m':
                    g[
                        'pre'
                    ].tolist(),

                'duration_sec':
                    (
                        restore_duration
                        + pre_duration
                    ),

                'max_segment_joint_delta_rad':
                    max(
                        restore_delta,
                        pre_delta,
                    ),

                'max_total_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_pre
                                - q_supported
                            )
                        )
                    ),

                'ik_position_error_m':
                    position_error,

                'ik_orientation_error_rad':
                    orientation_error,

                'collision_screen': {
                    'passed':
                        True,

                    'minimum_gap_m':
                        min(
                            restore_min_gap,
                            pre_gap,
                        ),

                    'known_fast67_restore_path_rescreened':
                        True,

                    'fresh_preplace_path_screened':
                        True,

                    'carried_book_screened':
                        True,
                },
            }

            elapsed = (
                time.perf_counter()
                - started
            )

            print(
                '[DAY8 FAST PREPLACE V2][PASS] '
                f'bias={bias:.2f} '
                f'wall={elapsed:.3f}s '
                f'restore_delta={restore_delta:.3f} '
                f'pre_delta={pre_delta:.3f}',
                flush=True,
            )

            return {
                'passed':
                    True,

                'selected_arm':
                    arm,

                'joint_names':
                    names,

                'target_rotation':
                    R.tolist(),

                'tip_T_book':
                    Tattach.tolist(),

                'pre_place_xyz_m':
                    g[
                        'pre'
                    ].tolist(),

                'lower_place_xyz_m':
                    g[
                        'low'
                    ].tolist(),

                'retract_xyz_m':
                    g[
                        'retract'
                    ].tolist(),

                'segments': {
                    'pre_place':
                        segment,
                },

                'bin_scene':
                    scene,

                'initial_joint_positions':
                    dict(joints),

                'collision_meshes_loaded':
                    envelopes.mesh_count,

                'baseline_housing_envelope_overlaps':
                    len(
                        envelopes.initial_pairs
                    ),

                'book_dimensions_m':
                    [.16, .02, .25],

                'planned_controlled_lowering_m':
                    g[
                        'lowering_m'
                    ],

                'book_uncertainty_margin_m':
                    settings[
                        'book_margin_m'
                    ],

                'placement_bias_toward_reachable_edge':
                    bias,

                'fast_staged_planning':
                    True,

                'known_restore_waypoint_used':
                    True,

                'preplace_planning_wall_sec':
                    elapsed,
            }

        except GeometryError as exc:
            failures.append({
                'bias':
                    bias,

                'reason':
                    str(exc),
            })

            print(
                '[DAY8 FAST PREPLACE V2][REJECT] '
                f'bias={bias:.2f} '
                f'reason={exc}',
                flush=True,
            )

            if (
                'BUDGET_EXCEEDED'
                in str(exc)
            ):
                break

    elapsed = (
        time.perf_counter()
        - started
    )

    raise GeometryError(
        'FAST_PREPLACE_V2_FAILED: '
        f'wall={elapsed:.3f}s '
        f'failures={failures}'
    )


# ============================================================
# DAY8 FAST PREPLACE V2
#
# Do NOT ask IK to rediscover the 90-degree carry restoration.
#
# Waypoint 1:
#   exact compact-carry joints already solved/screened by FAST67
#
# Waypoint 2:
#   fresh bin-relative pre-place IK from that validated pose
#
# Both are sent as ONE trajectory.  There is no upright dwell.
# ============================================================

def build_fast_preplace_plan(
    model,
    envelopes,
    joints,
    arm,
    R,
    Tattach,
    scene,
    settings,
):
    import time

    started = time.perf_counter()

    budget = float(
        settings.get(
            'fast_preplace_budget_sec',
            4.0,
        )
    )

    deadline = started + budget

    names = [
        f'arm_{arm}_{i}_joint'
        for i in range(1, 8)
    ]

    chain = model.chain(
        'base_footprint',
        f'gripper_{arm}_grasping_link',
    )

    q_supported = np.asarray(
        [
            joints[n]
            for n in names
        ],
        dtype=float,
    )

    restore_raw = settings.get(
        'known_restore_positions_rad'
    )

    if (
        not isinstance(
            restore_raw,
            (list, tuple),
        )
        or len(restore_raw) != 7
    ):
        raise GeometryError(
            'KNOWN_COMPACT_CARRY_JOINTS_MISSING'
        )

    q_restore = np.asarray(
        restore_raw,
        dtype=float,
    )

    T_supported = model.forward(
        chain,
        joints,
    ).tip_transform

    # --------------------------------------------------------
    # 1. Re-screen the KNOWN reverse support path in the
    # CURRENT measured Day8 scene.
    # --------------------------------------------------------

    envelopes.set_start(
        joints
    )

    envelopes.set_book_start(
        joints,
        Tattach,
    )

    fixed = dict(joints)

    restore_min_gap = float(
        'inf'
    )

    restore_samples = 9

    for sample in interpolate_joint_path(
        q_supported,
        q_restore,
        samples=restore_samples,
    ):
        if time.perf_counter() >= deadline:
            raise GeometryError(
                'FAST_PREPLACE_BUDGET_EXCEEDED_DURING_RESTORE_SCREEN'
            )

        jq = dict(fixed)

        jq.update(
            zip(
                names,
                sample,
            )
        )

        report = envelopes.check(
            jq,
            scene,
            Tattach,
            True,
            None,
            settings[
                'robot_margin_m'
            ],
            settings[
                'book_margin_m'
            ],
        )

        if not report['passed']:
            raise GeometryError(
                'KNOWN_RESTORE_PATH_UNSAFE:'
                + ','.join(
                    report[
                        'violations'
                    ][:5]
                )
            )

        restore_min_gap = min(
            restore_min_gap,
            report[
                'minimum_environment_envelope_gap_m'
            ],
        )

    restore_joints = dict(
        fixed
    )

    restore_joints.update(
        zip(
            names,
            q_restore,
        )
    )

    T_restore = model.forward(
        chain,
        restore_joints,
    ).tip_transform

    restore_orientation_error = float(
        np.linalg.norm(
            rotation_vector(
                R
                @ T_restore[
                    :3,
                    :3
                ].T
            )
        )
    )

    if restore_orientation_error > .10:
        raise GeometryError(
            'KNOWN_COMPACT_CARRY_ORIENTATION_MISMATCH:'
            + str(
                restore_orientation_error
            )
        )

    restore_delta = float(
        np.max(
            np.abs(
                q_restore
                - q_supported
            )
        )
    )

    restore_tip_distance = float(
        np.linalg.norm(
            T_restore[:3, 3]
            - T_supported[:3, 3]
        )
    )

    restore_duration = max(
        .20,
        restore_tip_distance
        / settings[
            'preplace_speed_mps'
        ],
        restore_delta
        / settings[
            'max_joint_speed_radps'
        ],
    )

    restore_record = {
        'positions':
            q_restore.tolist(),

        'dt_sec':
            restore_duration,

        'time_sec':
            restore_duration,

        'target_xyz_m':
            T_restore[
                :3,
                3
            ].tolist(),

        'ik_position_error_m':
            0.0,

        'ik_orientation_error_rad':
            restore_orientation_error,

        'known_fast67_waypoint':
            True,
    }

    # --------------------------------------------------------
    # 2. Solve only the remaining compact-carry -> pre-place
    # move. Starting orientation now already matches R.
    # --------------------------------------------------------

    failures = []

    for bias in (
        0.0,
        0.50,
        0.75,
    ):

        if time.perf_counter() >= deadline:
            break

        try:
            g = _fast_stage_geometry(
                model,
                joints,
                arm,
                R,
                Tattach,
                scene,
                settings,
                bias,
            )

            stage_settings = dict(
                settings
            )

            # More realistic than the previous 45-iteration
            # zero-retry solve, while still strictly bounded.
            stage_settings[
                'fast_plan_ik_iterations'
            ] = 110

            stage_settings[
                'fast_plan_joint_samples'
            ] = 8

            stage_settings[
                'fast_stage_seed_retries'
            ] = 1

            (
                q_pre,
                pre_record,
                pre_gap,
                position_error,
                orientation_error,
            ) = _fast_day8_solve(
                model,
                envelopes,
                chain,
                names,
                fixed,
                q_restore,
                T_restore[:3, 3],
                g['pre'],
                R,
                scene,
                Tattach,
                stage_settings,
                True,
                None,
                1.40,
                deadline,
            )

            pre_delta = float(
                np.max(
                    np.abs(
                        q_pre
                        - q_restore
                    )
                )
            )

            pre_distance = float(
                np.linalg.norm(
                    g['pre']
                    - T_restore[
                        :3,
                        3
                    ]
                )
            )

            pre_duration = max(
                .20,
                pre_distance
                / settings[
                    'preplace_speed_mps'
                ],
                pre_delta
                / settings[
                    'max_joint_speed_radps'
                ],
            )

            pre_record[
                'dt_sec'
            ] = pre_duration

            # IMPORTANT:
            # cumulative controller time.
            # Robot passes through q_restore and continues
            # directly to q_pre without waiting there.
            pre_record[
                'time_sec'
            ] = (
                restore_duration
                + pre_duration
            )

            segment = {
                'passed':
                    True,

                'waypoints': [
                    restore_record,
                    pre_record,
                ],

                'start_positions_rad':
                    q_supported.tolist(),

                'start_xyz_m':
                    T_supported[
                        :3,
                        3
                    ].tolist(),

                'target_xyz_m':
                    g[
                        'pre'
                    ].tolist(),

                'duration_sec':
                    (
                        restore_duration
                        + pre_duration
                    ),

                'max_segment_joint_delta_rad':
                    max(
                        restore_delta,
                        pre_delta,
                    ),

                'max_total_joint_delta_rad':
                    float(
                        np.max(
                            np.abs(
                                q_pre
                                - q_supported
                            )
                        )
                    ),

                'ik_position_error_m':
                    position_error,

                'ik_orientation_error_rad':
                    orientation_error,

                'collision_screen': {
                    'passed':
                        True,

                    'minimum_gap_m':
                        min(
                            restore_min_gap,
                            pre_gap,
                        ),

                    'known_fast67_restore_path_rescreened':
                        True,

                    'fresh_preplace_path_screened':
                        True,

                    'carried_book_screened':
                        True,
                },
            }

            elapsed = (
                time.perf_counter()
                - started
            )

            print(
                '[DAY8 FAST PREPLACE V2][PASS] '
                f'bias={bias:.2f} '
                f'wall={elapsed:.3f}s '
                f'restore_delta={restore_delta:.3f} '
                f'pre_delta={pre_delta:.3f}',
                flush=True,
            )

            return {
                'passed':
                    True,

                'selected_arm':
                    arm,

                'joint_names':
                    names,

                'target_rotation':
                    R.tolist(),

                'tip_T_book':
                    Tattach.tolist(),

                'pre_place_xyz_m':
                    g[
                        'pre'
                    ].tolist(),

                'lower_place_xyz_m':
                    g[
                        'low'
                    ].tolist(),

                'retract_xyz_m':
                    g[
                        'retract'
                    ].tolist(),

                'segments': {
                    'pre_place':
                        segment,
                },

                'bin_scene':
                    scene,

                'initial_joint_positions':
                    dict(joints),

                'collision_meshes_loaded':
                    envelopes.mesh_count,

                'baseline_housing_envelope_overlaps':
                    len(
                        envelopes.initial_pairs
                    ),

                'book_dimensions_m':
                    [.16, .02, .25],

                'planned_controlled_lowering_m':
                    g[
                        'lowering_m'
                    ],

                'book_uncertainty_margin_m':
                    settings[
                        'book_margin_m'
                    ],

                'placement_bias_toward_reachable_edge':
                    bias,

                'fast_staged_planning':
                    True,

                'known_restore_waypoint_used':
                    True,

                'preplace_planning_wall_sec':
                    elapsed,
            }

        except GeometryError as exc:
            failures.append({
                'bias':
                    bias,

                'reason':
                    str(exc),
            })

            print(
                '[DAY8 FAST PREPLACE V2][REJECT] '
                f'bias={bias:.2f} '
                f'reason={exc}',
                flush=True,
            )

            if (
                'BUDGET_EXCEEDED'
                in str(exc)
            ):
                break

    elapsed = (
        time.perf_counter()
        - started
    )

    raise GeometryError(
        'FAST_PREPLACE_V2_FAILED: '
        f'wall={elapsed:.3f}s '
        f'failures={failures}'
    )
