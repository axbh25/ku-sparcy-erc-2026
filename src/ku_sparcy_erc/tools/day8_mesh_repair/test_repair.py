#!/usr/bin/env python3
"""Offline regression: algebra, mesh predicates, full replay, denser path check.
No ROS imports, no robot commands. The input archives are never modified.
"""
import argparse,json,math,sys,tarfile,tempfile,time
from pathlib import Path
import numpy as np
from replay_scene import load_inputs

def unpack(archive,dest):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            target=(dest/member.name).resolve()
            if dest!=target and dest not in target.parents:raise ValueError('Unsafe archive path')
            if not (member.isdir() or member.isfile()):raise ValueError('Links/devices not accepted')
        # All members were checked above; works on Python 3.10 as used by Humble.
        tf.extractall(dest)


def run_tests(d,output):
    core=d['core'];screen=d['screen'];model=d['model'];fixed=dict(d['joints']);names=screen.names
    from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel,rotation_z
    from ku_sparcy_erc.day8_geometry import colour_mask,transform,stl_triangles
    from ku_sparcy_erc.day8_planner import sat_gap
    original=URDFKinematicModel.from_file(str(d['repro']/'packages/erc_description/urdf/tiago_pro.urdf'))
    checks=[];details={};rng=np.random.default_rng(5731)
    def add(name,ok,**kwargs):
        checks.append({'name':name,'passed':bool(ok),**kwargs});print(('PASS ' if ok else 'FAIL ')+name,flush=True)
    q0=np.array([fixed[n] for n in names]);lo,hi=model.limits(names,.025)
    maxfk=0.
    for _ in range(35):
        q=rng.uniform(lo,hi);j=core.joint_values(fixed,names,q)
        A=original.forward(screen.chain,j).tip_transform;B=model.forward(screen.chain,j).tip_transform
        maxfk=max(maxfk,float(abs(A-B).max()))
    add('Cached FK matches captured implementation',maxfk<1e-10,max_transform_error=maxfk)
    q=np.clip(q0,lo+1e-4,hi-1e-4);T,J=model.tip_jacobian(screen.chain,names,q,fixed);finite=np.empty_like(J)
    for i in range(7):
        x=q.copy();x[i]+=1e-7;U=model.forward(screen.chain,core.joint_values(fixed,names,x)).tip_transform
        finite[:,i]=np.r_[(U[:3,3]-T[:3,3])/1e-7,(.25/math.sqrt(2)*(U[:3,:3]-T[:3,:3])/1e-7).ravel()]
    je=float(abs(J-finite).max());add('Analytic Jacobian finite-difference check',je<1e-6,max_error=je)
    As=[];Bs=[];ha=[];hb=[]
    from ku_sparcy_erc.urdf_kinematics import rpy_matrix
    for _ in range(500):
        As.append(transform(rpy_matrix(rng.normal(size=3)),rng.normal(size=3)*.3));Bs.append(transform(rpy_matrix(rng.normal(size=3)),rng.normal(size=3)*.3))
        ha.append(rng.uniform(.01,.35,3));hb.append(rng.uniform(.01,.35,3))
    bat=core.sat_many(np.array(As),np.array(ha),np.array(Bs),np.array(hb));scalar=np.array([sat_gap(a,h,b,k) for a,h,b,k in zip(As,ha,Bs,hb)])
    se=float(abs(bat-scalar).max());add('Vectorized SAT matches captured scalar SAT',se<1e-9,max_gap_difference_m=se)
    tr=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]);pts=np.array([[.2,.2,.5],[1,1,0],[-1,-1,0]])
    nearest,_,distance=core.closest_triangles(pts,tr)
    add('Point/triangle face-edge-vertex tests',np.allclose(nearest,[[.2,.2,0],[.5,.5,0],[0,0,0]],atol=1e-10))
    add('Triangle/box rejects actual intersection',core.triangle_box_intersects(transform(t=[.2,.2,0]),np.array([.1]*3),tr))
    add('Triangle/box accepts separated box',not core.triangle_box_intersects(transform(t=[.2,.2,1]),np.array([.1]*3),tr))
    assets=d['assets'];solid=[0,0,-.205];empty=[0,0,-.08]
    add('Bin shell containment versus hollow interior',core.point_inside_mesh(np.array(solid),assets.bin_tri)
        and not core.point_inside_mesh(np.array(empty),assets.bin_tri))
    tv=stl_triangles(assets.table_path)
    meshvol=abs(float(np.einsum('ij,ij->i',tv[:,0],np.cross(tv[:,1],tv[:,2])).sum()/6))
    boxvol=sum(float(np.prod(2*h)) for _,h in assets.table_boxes)
    add('Five table solids reproduce signed mesh volume',abs(meshvol-boxvol)<1e-7,mesh_volume_m3=meshvol,box_volume_m3=boxvol)
    expired=False
    try:core.solve_endpoint(model,screen.chain,names,fixed,np.zeros(3),np.eye(3),q0,time.monotonic()-1)
    except Exception as e:expired='BUDGET_EXCEEDED' in str(e)
    add('Expired solver budget fails closed',expired)
    add('Latest image cannot certify a held blue book',int(colour_mask(d['rgb'],'blue').sum())==0,blue_pixels=int(colour_mask(d['rgb'],'blue').sum()))
    start=time.monotonic();plan=core.build_complete_plan(screen,fixed,d['restore'],d['rotation'],d['scene'],d['settings'],start+30.)
    elapsed=time.monotonic()-start
    add('Full candidate dock/pre-place/lower/open/retract plan',plan['passed'] and 0<plan['docking_forward_m']<=.22,planning_wall_sec=elapsed)
    # New observed dock state must produce another complete plan at zero extra dock.
    start=time.monotonic();post=core.build_complete_plan(screen,fixed,d['restore'],d['rotation'],plan['bin_scene'],d['settings'],start+30.,offsets=(0.,))
    add('Replan at achieved dock requires no additional shift',post['docking_forward_m']==0,planning_wall_sec=time.monotonic()-start)
    for name,seg in plan['segments'].items():
        add(name+' original IK tolerances',seg['ik_position_error_m']<=.0025 and seg['ik_orientation_error_rad']<=.025)
        times=[w['time_sec'] for w in seg['waypoints']]
        add(name+' monotonic trajectory timestamps',all(a<b for a,b in zip([0.]+times,times)))
    last=plan['segments']['lower_place']['waypoints'][-1]['positions'];jl=core.joint_values(fixed,names,last)
    deposited=model.forward(screen.chain,jl).tip_transform@screen.attach
    # Independent 4x-denser joint interpolation than the planner's .025-rad grid.
    count=0;bad=[];q=np.array([fixed[n] for n in names]);start=time.monotonic()
    for stage in ('pre_place','lower_place','retract'):
        if stage=='retract':fixed[f'gripper_{screen.arm}_finger_joint']=.04
        for w in plan['segments'][stage]['waypoints']:
            nxt=np.asarray(w['positions']);n=max(3,math.ceil(float(abs(nxt-q).max())/.00625))+1
            for u in np.linspace(0,1,n):
                report=screen.check(core.joint_values(fixed,names,q+u*(nxt-q)),plan['bin_scene'],carried=stage!='retract',deposited=deposited if stage=='retract' else None)
                count+=1
                if not report['passed']:bad.append((stage,report['violations']))
            q=nxt
    add('Denser complete arm-path collision rescreen',not bad,samples=count,wall_sec=time.monotonic()-start,failures=bad[:5])
    # The checker must still reject a real geometry violation, not just allow
    # every target. Shift the tabletop directly through a known gripper pose.
    wrong=json.loads(json.dumps(plan['bin_scene']));M=np.asarray(wrong['base_T_table_top']);tip=model.forward(screen.chain,jl).tip_transform
    M[:3,3]=tip[:3,3];wrong['base_T_table_top']=M.tolist()
    reject=screen.check(jl,wrong)
    add('Deliberate table collision is rejected',not reject['passed'] and any('TABLE' in s for s in reject['violations']))
    details.update(checks=checks,passed=all(x['passed'] for x in checks),nominal_plan=plan,
        current_rgb_blue_pixels=int(colour_mask(d['rgb'],'blue').sum()),
        limitations=['Offline model/sensor replay, NOT a Gazebo execution.',
        'Later RGB-D is not the historical grasp snapshot; a retained book is assumed for the hypothetical planned trajectory.',
        'Sampled inflated-envelope collision checking is not a continuous collision certificate.',
        'Benchmark wall time belongs to this machine, not the user laptop.',
        'Physical support, release, sustained grasp, controller interpolation configuration and actual deposit still need live verification.'])
    Path(output).write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
    return details


def main():
    p=argparse.ArgumentParser();p.add_argument('--repro-archive',required=True);p.add_argument('--scene-archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='day8_mesh_test_') as temp:
        unpack(a.repro_archive,temp);unpack(a.scene_archive,temp)
        d=load_inputs(Path(temp)/'day8_repro',Path(temp)/'day8_scene');result=run_tests(d,a.output)
    print('[OFFLINE MESH REPAIR]['+('PASS' if result['passed'] else 'FAIL')+']')
    raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
