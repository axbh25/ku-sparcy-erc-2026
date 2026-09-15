#!/usr/bin/env python3
"""Reject ambiguous deposit, stale evidence, premature release, or partial plans."""
import argparse
import json
import math
import cv2
from pathlib import Path


def validate(d,mode):
    checks=[]
    def check(label,condition):checks.append((label,bool(condition)))
    def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
    check('DAY7 INPUT',d.get('day7_input_passed') is True)
    check('RESULT STATUS',d.get('passed') is True and d.get('state')=='DONE' and d.get('operation')==mode)
    check('SELECTED ARM',d.get('selected_arm') in ('left','right') and d.get('selected_arm')==d.get('day5_selected_arm'))
    check('SOURCE EPOCH',finite(d.get('simulation_time_sec')) and finite(d.get('source_day7_sim_time'))
          and d['simulation_time_sec']>=d['source_day7_sim_time'])
    check('BASE STATIONARY',finite(d.get('max_base_drift_m')) and d['max_base_drift_m']<=.025
          and finite(d.get('max_base_yaw_drift_deg')) and d['max_base_yaw_drift_deg']<=1.5)
    counts=d.get('command_counts') or {}
    check('NO BASE MOTION',counts.get('base_nonzero')==0)
    check('PUBLIC POSITION ONLY',d.get('gripper_command_mode')=='position'
          and d.get('gripper_public_topic')==f"/gripper_{d.get('selected_arm')}_controller/joint_trajectory"
          and d.get('effort_commanded') is False and d.get('raw_topic_commanded') is False
          and d.get('torso_commanded') is False)
    check('NO HIDDEN ORACLE',d.get('hidden_simulator_oracle_used') is False and d.get('world_pose_queried') is False)
    scene=d.get('bin_geometry') or {};obs=d.get('held_book_observation') or {}
    check('LIVE BIN GEOMETRY',scene.get('live_geometry') is True and scene.get('confirming_frames',0)>=3
          and len(scene.get('rim_edge_support_counts',[]))==4
          and min(scene.get('rim_edge_support_counts',[0]))>=7
          and scene.get('floor_source') in ('live_depth_plane','live_rim_minus_official_mesh_interior_depth'))
    check('TIMESTAMP PAIRING',finite(scene.get('rgb_depth_skew_sec')) and 0<=scene['rgb_depth_skew_sec']<=.20)
    check('HELD OBJECT OBSERVED',obs.get('consistent_colour_points',0)>=20
          and finite(obs.get('surface_residual_m')) and obs['surface_residual_m']<=.025)
    ledger=d.get('contact_evidence') or {}
    check('CONTACT IDENTITY',isinstance(ledger.get('held_object_contact_id'),str)
          and bool(ledger.get('held_object_contact_id')))
    check('NO UNINTENDED CONTACT',ledger.get('robot_bin_contacts')==0
          and ledger.get('unexpected_contacts')==0 and ledger.get('premature_book_bin_contacts')==0)
    if mode=='survey':
        check('SURVEY DOES NOT MOVE ARM',counts.get('arm')==0)
    if mode=='plan':
        check('PLAN COMMANDS NOTHING',all(counts.get(k)==0 for k in ('base_zero','base_nonzero','head','arm','gripper')))
    if mode in ('plan','execute'):
        plan=d.get('plan') or {};segments=plan.get('segments') or {}
        check('PLACEMENT PLAN',plan.get('passed') is True and plan.get('collision_meshes_loaded',0)>0)
        for stage in ('pre_place','lower_place','retract'):
            s=segments.get(stage) or {};screen=s.get('collision_screen') or {}
            check(stage.upper()+' IK',s.get('passed') is True and bool(s.get('waypoints'))
                  and finite(s.get('ik_position_error_m')) and s['ik_position_error_m']<=.0025
                  and finite(s.get('ik_orientation_error_rad')) and s['ik_orientation_error_rad']<=.025)
            check(stage.upper()+' COLLISION SCREEN',screen.get('passed') is True
                  and screen.get('sampled_mesh_bounds') is True
                  and (screen.get('deposited_book_screened') if stage=='retract' else screen.get('carried_book_screened')) is True)
    if mode=='execute':
        done=d.get('completed_stages',[])
        check('ALL PLACEMENT STAGES',all(s in done for s in ('pre_place','lower_place','open','retract','deposit')))
        seen={item.get('stage') for item in d.get('bin_live_rechecks',[]) if item.get('passed') is True
              and item.get('surface_residual80_m',1) <= .025 and item.get('points',0)>=80}
        check('BIN GEOMETRY RECHECKED',{'pre_place','lower','release'} <= seen)
        first=d.get('first_book_bin_contact_sim_sec');start=d.get('release_started_sim_sec')
        opened=d.get('release_complete_sim_sec');retracted=d.get('retract_completed_sim_sec');now=d.get('simulation_time_sec')
        check('SUPPORT BEFORE RELEASE',d.get('support_verified_before_open') is True
              and all(finite(x) for x in (first,start,opened,retracted,now)) and first<start<=opened<retracted<now)
        seq=d.get('release_trace') or [];positions=[x.get('commanded') for x in seq]
        check('STEPWISE OPENING',len(seq)>=4 and all(finite(x) for x in positions)
              and positions==sorted(set(positions)) and max(positions)<=.069
              and all(abs(x['actual']-x['commanded'])<=.007 for x in seq))
        times=[item.get('time') for item in seq]
        check('SLOW OPENING TIMESTAMPS',len(times)>=4 and all(finite(t) for t in times)
              and finite(start) and finite(opened) and times[0]-start>=.79
              and all(b-a>=.79 for a,b in zip(times,times[1:])) and times[-1]<=opened+.01)
        check('OPEN GRIPPER',finite(d.get('gripper_open_actual')) and abs(d['gripper_open_actual']-.040)<=.007)
        support=[t for t in ledger.get('support_timestamps',[]) if finite(t) and finite(retracted) and t>=retracted]
        check('SUPPORT AFTER RETRACT',len(support)>=5 and support[-1]-support[0]>=2.
              and now-support[-1]<=.45 and max([b-a for a,b in zip(support,support[1:])],default=0)<=.45)
        tip=ledger.get('last_selected_fingertip_time')
        check('FINGERS DISENGAGED',finite(tip) and finite(start) and tip>=start-.8 and now-tip>=.8)
        check('RELEASE AND DEPOSIT VERIFIED',d.get('release_verified') is True and d.get('deposit_verified') is True)
        check('CONTROLLED PLACEMENT CONTACT',d.get('book_bin_depth_samples',0)>0
              and finite(d.get('maximum_book_bin_penetration_m'))
              and d['maximum_book_bin_penetration_m']<=.006)
        rescreen=d.get('actual_retract_rescreen') or {}
        check('ACTUAL RETRACT PATH RESCREENED',rescreen.get('passed') is True
              and rescreen.get('actual_deposited_book_checked') is True and rescreen.get('samples',0)>0)
        check('CONTACT STREAMS LIVE AFTER RELEASE',all(finite(d.get(k)) and now-d[k]<=.8
              for k in ('last_robot_contact_stream_sim_sec','last_bin_contact_stream_sim_sec')))
        v=d.get('visual_deposit_observations',[])
        current_v=[x for x in v if finite(x.get('stamp')) and finite(retracted) and retracted<=x['stamp']<=now
                   and x.get('points',0)>=16]
        visual_ok=len({x['stamp'] for x in current_v})>=3 and now-current_v[-1]['stamp']<=.55
        check('POST-RETRACT VISUAL EVIDENCE OR RED-ID EVIDENCE',
              visual_ok or (d.get('target_colour')=='red' and d.get('red_target_visual_limitation') is True and len(support)>=5))
        deposited_image=(d.get('evidence_timestamps') or {}).get('deposited') or {}
        image_stamp=deposited_image.get('rgb_stamp_sec')
        check('FRESH POST-RETRACT IMAGE',finite(image_stamp) and finite(retracted)
              and image_stamp>=retracted and now-image_stamp<=.55)
        check('PLAN AND LIVE EVIDENCE',all(k in d.get('evidence_images',{}) for k in ('plan','pre_place','supported','released','retract','deposited')))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--mode',choices=['survey','plan','execute'],default='execute');a=p.parse_args()
    try:
        d=json.loads(Path(a.result).read_text());checks=validate(d,a.mode)
    except Exception as exc:
        print('[DAY8 VALIDATOR][FAIL] '+str(exc));raise SystemExit(1)
    for label,passed in checks:print('[DAY8 '+label+']['+('PASS' if passed else 'FAIL')+']')
    image_ok=True
    for label,path in (d.get('evidence_images') or {}).items():
        ok=Path(path).is_file() and Path(path).stat().st_size>100 and cv2.imread(path) is not None
        print('[DAY8 IMAGE '+label+']['+('PASS' if ok else 'FAIL')+']');image_ok &= ok
    ok=all(x[1] for x in checks) and image_ok
    marker='DAY8 PLACEMENT PLAN' if a.mode=='plan' else ('DAY8 ACCEPTANCE' if a.mode=='execute' else 'DAY8 SURVEY')
    print('['+marker+']['+('PASS' if ok else 'FAIL')+']')
    raise SystemExit(0 if ok else 1)

if __name__=='__main__':main()
