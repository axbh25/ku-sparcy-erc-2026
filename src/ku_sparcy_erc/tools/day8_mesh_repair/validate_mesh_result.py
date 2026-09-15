#!/usr/bin/env python3
"""Audit deliberate docking separately; retain original placement interlocks."""
import argparse,copy,importlib.util,json,math,os
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--team',default=os.environ.get('KU_SPARCY_TEAM','/opt/erc_ws/src/ku_sparcy_erc'));a=p.parse_args()
    data=json.loads(Path(a.result).read_text());repair=data.get('mesh_repair') or {};checks=[]
    def add(k,v):checks.append((k,bool(v)))
    add('REPAIR VERSION',repair.get('version')=='measured_mesh_repair_1')
    docking=repair.get('docking');counts=repair.get('placement_phase_command_counts')
    if docking:
        add('EXPLICIT DOCK AUTHORIZATION',repair.get('docking_authorized') is True)
        add('DOCK RESULT',docking.get('passed') is True)
        target=docking.get('target_forward_m');actual=docking.get('actual_forward_m')
        add('BOUNDED FORWARD DOCK',isinstance(target,(int,float)) and 0<target<=.22
            and isinstance(actual,(int,float)) and abs(actual-target)<=.005)
        trace=docking.get('trace') or []
        add('DOCK COMMAND/ODOM EVIDENCE',len(trace)>=5 and all(
            0<=x['command_vx_mps']<=.050001 and abs(x['cross_m'])<=.00401
            and abs(x['yaw_rad'])<=math.radians(.2501) and x['arm_max_delta_rad']<=.02501
            and 0<=x['sim']-x['scan_stamp']<=.6001
            and (x['lidar_clearance_m']>=.55 or x['lidar_cluster_points']<3) for x in trace))
        held=docking.get('held_book_checks') or []
        add('HELD BOOK RECHECKED DURING DOCK',len(held)>=2 and all(x.get('consistent_colour_points',0)>=20
            and x.get('surface_residual_m',1)<=.025 and abs(x['rgb_stamp']-x['depth_stamp'])<=.20 for x in held))
        add('DOCK SWEPT MESH CHECK',len(docking.get('planned_sweep_samples_m',[]))>=2)
    else:
        add('NO UNREPORTED BASE MOVEMENT',data.get('command_counts',{}).get('base_nonzero')==0)
    add('PLACEMENT BASE STATIONARY',repair.get('placement_max_base_drift_m',1)<=.025
        and repair.get('placement_max_base_yaw_drift_deg',99)<=1.5)
    add('PLACEMENT ZERO NONZERO BASE COMMANDS',isinstance(counts,dict) and counts.get('base_nonzero')==0)
    scene=data.get('bin_geometry') or {};reg=scene.get('bin_mesh_registration') or {};table=scene.get('table_measurement') or {}
    add('LIVE REGISTERED BIN/TABLE',scene.get('repair_geometry_version')=='measured_mesh_repair_1'
        and scene.get('fresh_surface_registration') is True and scene.get('confirming_frames',0)>=3
        and reg.get('points',0)>=150 and reg.get('median_m',1)<=.004 and reg.get('p80_m',1)<=.008
        and table.get('points',0)>=250)
    spec=importlib.util.spec_from_file_location('legacy_day8_validator',str(Path(a.team)/'tools/validate_day8_result.py'))
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    # These three predicates belonged to the old whole-Day8 zero-base/full-rim
    # geometry contract; the explicit new predicates above replace them.
    replace={'BASE STATIONARY','NO BASE MOTION','LIVE BIN GEOMETRY'}
    for name,ok in legacy.validate(data,'execute'):
        if name not in replace:checks.append(('PLACEMENT '+name,ok))
    for name,ok in checks:print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    image_ok=True
    for name,path in (data.get('evidence_images') or {}).items():
        good=Path(path).is_file() and legacy.cv2.imread(path) is not None
        image_ok &= good;print(f"[{'PASS' if good else 'FAIL'}] EVIDENCE IMAGE {name}")
    good=all(v for _,v in checks) and image_ok
    print('[DAY8 MESH REPAIR ACCEPTANCE]['+('PASS' if good else 'FAIL')+']')
    raise SystemExit(0 if good else 1)
if __name__=='__main__':main()
