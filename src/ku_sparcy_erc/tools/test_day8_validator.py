#!/usr/bin/env python3
"""Mutation tests: partial, premature, stale, and table-only results must fail."""
import copy
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('v',Path(__file__).with_name('validate_day8_result.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
segment=dict(passed=True,waypoints=[{'positions':[0]*7}],ik_position_error_m=.001,ik_orientation_error_rad=.01,
             collision_screen=dict(passed=True,sampled_mesh_bounds=True,carried_book_screened=True,deposited_book_screened=True))
d=dict(day7_input_passed=True,passed=True,state='DONE',operation='execute',selected_arm='right',day5_selected_arm='right',
       simulation_time_sec=13.,source_day7_sim_time=1.,max_base_drift_m=.001,max_base_yaw_drift_deg=.01,
       command_counts=dict(base_zero=100,base_nonzero=0,arm=20,gripper=90,head=0),
       gripper_command_mode='position',gripper_public_topic='/gripper_right_controller/joint_trajectory',
       effort_commanded=False,raw_topic_commanded=False,torso_commanded=False,hidden_simulator_oracle_used=False,world_pose_queried=False,
       bin_geometry=dict(live_geometry=True,confirming_frames=3,rim_edge_support_counts=[20]*4,
                         floor_source='live_depth_plane',rgb_depth_skew_sec=.03),
       held_book_observation=dict(consistent_colour_points=80,surface_residual_m=.006),
       contact_evidence=dict(held_object_contact_id='opaque_object',robot_bin_contacts=0,unexpected_contacts=0,premature_book_bin_contacts=0,
                             support_timestamps=[round(2+i*.1,3) for i in range(111)],last_selected_fingertip_time=3.4),
       plan=dict(passed=True,collision_meshes_loaded=15,segments={k:copy.deepcopy(segment) for k in ('pre_place','lower_place','retract')}),
       completed_stages=['pre_place','lower_place','open','retract','deposit'],support_verified_before_open=True,
       first_book_bin_contact_sim_sec=2.,release_started_sim_sec=3.,release_complete_sim_sec=6.25,retract_completed_sim_sec=10.,
       release_trace=[dict(time=3.8+i*.8,commanded=x,actual=x) for i,x in enumerate([.018,.026,.034,.04])],
       gripper_open_actual=.04,release_verified=True,deposit_verified=True,book_bin_depth_samples=20,maximum_book_bin_penetration_m=.002,
       actual_retract_rescreen=dict(passed=True,actual_deposited_book_checked=True,samples=80),
       last_robot_contact_stream_sim_sec=12.99,last_bin_contact_stream_sim_sec=12.99,
       target_colour='blue',visual_deposit_observations=[dict(stamp=t,points=30) for t in (12.8,12.9,13.)],
       evidence_images={k:'not_read_by_pure_validator' for k in ('plan','pre_place','supported','released','retract','deposited')},
       evidence_timestamps=dict(deposited=dict(rgb_stamp_sec=12.95)),
       bin_live_rechecks=[dict(stage=k,passed=True,points=200,surface_residual80_m=.008) for k in ('pre_place','lower','release')])


def valid(x):return all(ok for _,ok in v.validate(x,'execute'))

assert valid(d),v.validate(d,'execute')
for label,mutate in [
    ('table contact alone',lambda x:x['contact_evidence'].update(support_timestamps=[])),
    ('robot-bin contact',lambda x:x['contact_evidence'].update(robot_bin_contacts=1)),
    ('premature release',lambda x:x.update(release_started_sim_sec=1.)),
    ('missing retract',lambda x:x['completed_stages'].remove('retract')),
    ('closed gripper',lambda x:x.update(gripper_open_actual=.01)),
    ('contact stream starvation',lambda x:x.update(last_robot_contact_stream_sim_sec=8.)),
    ('stale visual deposit',lambda x:x.update(visual_deposit_observations=[dict(stamp=5.,points=30)]*3)),
    ('instant open',lambda x:x.update(release_trace=[dict(time=3.1,commanded=.04,actual=.04)])),
    ('excess contact penetration',lambda x:x.update(maximum_book_bin_penetration_m=.03)),
    ('base motion',lambda x:x.update(max_base_drift_m=.10)),
]:
    bad=copy.deepcopy(d);mutate(bad)
    assert not valid(bad),label
    print('[REJECT '+label+'][PASS]')
print('[DAY8 VALIDATOR NEGATIVE TESTS][PASS]')
