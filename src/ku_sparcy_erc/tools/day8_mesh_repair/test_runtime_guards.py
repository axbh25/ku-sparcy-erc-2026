#!/usr/bin/env python3
"""Mocked message/state tests only. Not a ROS integration or physics test."""
import importlib,sys,time,types
from pathlib import Path
import numpy as np

def run(team):
    sys.path.insert(0,str(team));sys.path.insert(0,str(Path(__file__).parent))
    sys.modules['rclpy']=types.ModuleType('rclpy')
    class Twist:
        def __init__(self):self.linear=types.SimpleNamespace(x=0.,y=0.,z=0.);self.angular=types.SimpleNamespace(x=0.,y=0.,z=0.)
    msg=types.ModuleType('geometry_msgs.msg');msg.Twist=Twist
    sys.modules['geometry_msgs']=types.ModuleType('geometry_msgs');sys.modules['geometry_msgs.msg']=msg
    class Base:
        def log(self,text):pass
        def finish(self,ok,reason):self.done=True;self.passed=ok;self.reason=reason
        def keep_safe_hold(self):pass
        def transition(self,state,reason):self.state=state
        def request(self,x):self.state='WAIT_APPROVAL_'+x
        def _tick(self):self.delegated=True
    original=types.ModuleType('ku_sparcy_erc.day8_bin_place');original.Day8BinPlace=Base;original.DEFAULTS={}
    sys.modules['ku_sparcy_erc.day8_bin_place']=original
    module=importlib.import_module('day8_mesh_node');C=module.MeasuredMeshDay8
    class Pub:
        def __init__(self):self.messages=[]
        def publish(self,v):self.messages.append(v)
        def get_subscription_count(self):return 1
    class Future:
        def __init__(self,p):self.p=p
        def done(self):return True
        def result(self):return self.p
    def node():
        n=C.__new__(C);n.p=dict(original.DEFAULTS);n.p.update(joint_fresh_sec=.55,scan_fresh_sec=.6,
            clock_stall_wall_timeout_sec=10.,gripper_retention_max_position_m=.02)
        n.sim=20.;n.last_clock_wall=time.monotonic();n.joint_stamp=n.odom_stamp=n.scan_stamp=n.last_contacts_rx=n.sim
        n.fresh=lambda stamp,age:stamp is not None and 0<=n.sim-stamp<=age
        n.day7=dict(simulation_time_sec=10.,final_odom_xy=[0.,0.],final_yaw_rad=0.)
        n.odom=[0.,0.,0.];n.placement_anchor=None;n.dock_active=False;n.dock_done=False
        n.total_max_displacement=n.placement_max_drift=n.placement_max_yaw=n.max_drift=n.max_yaw_drift=0.
        n.initial_other={};n.first_book_bin_time=None;n.release_started=None;n.gripper_name='gripper_right_finger_joint'
        n.names=['a','b'];n.joints={'a':0.,'b':0.,n.gripper_name:0.};n.snapshot_joints=dict(n.joints)
        n.command_counts=dict(base_zero=0,base_nonzero=0);n.done=False;n.passed=False;n.cmd=Pub();n.op='execute'
        n.state='MESH_PLAN_COMPLETE';n.future=Future(dict(docking_forward_m=.18,planning_wall_sec=.1))
        n.count_publishers=lambda topic:1;n.scene_still_consistent=lambda name:None
        n.mesh_initial={};n.placement_counts_start=None;n.dock_trace=[];n.dock_v=0.
        n.front=types.SimpleNamespace(minimum_clearance_m=1.,hard_cluster_point_count=0)
        n.ledger=types.SimpleNamespace(tip_last=20.)
        n.clouds=lambda: (None,None,dict(consistent_colour_points=30,surface_residual_m=.001),n.sim,None)
        return n
    checks=[]
    def test(k,ok):checks.append((k,bool(ok)));print(('PASS ' if ok else 'FAIL ')+k)
    n=node();n._tick();test('No precision motion without explicit authorization',n.done and not n.passed and 'NOT_AUTHORIZED' in n.reason and not n.cmd.messages)
    n=node();n.p['repair_allow_precision_dock']=True;n.count_publishers=lambda x:2;n._tick();test('Competing base publisher blocks docking',n.done and 'EXCLUSIVE' in n.reason)
    n=node();n.op='plan';n._tick();test('Plan-only never sends a base command',n.passed and not n.cmd.messages)
    n=node()
    n.p['repair_allow_precision_dock']=True

    # Mock all live sensor timestamps required by the runtime node.
    n.rgb_stamp=n.sim
    n.depth_stamp=n.sim

    n._tick()

    n.sim+=.05

    n.joint_stamp=n.sim
    n.odom_stamp=n.sim
    n.scan_stamp=n.sim
    n.last_contacts_rx=n.sim
    n.rgb_stamp=n.sim
    n.depth_stamp=n.sim

    n._tick()
    test('Dock acceleration ramp starts at 0.002 m/s',abs(n.cmd.messages[-1].linear.x-.002)<1e-9)
    test('Dock never commands lateral or angular velocity',all(m.linear.y==0 and m.angular.z==0 for m in n.cmd.messages))
    n.odom[1]=.006;n._tick();test('Cross-track violation stops base',n.done and n.cmd.messages[-1].linear.x==0)
    n=node();n.p['repair_allow_precision_dock']=True;n._tick();n.front.minimum_clearance_m=.4;n.front.hard_cluster_point_count=3;n._tick();test('Front obstacle stops base',n.done and 'LIDAR' in n.reason and n.cmd.messages[-1].linear.x==0)
    n=node();n.p['repair_allow_precision_dock']=True;n._tick();n.joints['a']=.04;n._tick();test('Unexpected arm change during dock stops base',n.done and 'ARM_CHANGED' in n.reason)
    n=node();n.p['repair_allow_precision_dock']=True;n._tick();n.scan_stamp=1.;n._tick();test('Stale scan stops base',n.done and 'STALE_LIDAR'==n.reason)
    n=node();n.state='LOWERING';n._tick();test('Normal release/placement states delegate to inherited interlocks',getattr(n,'delegated',False))
    return checks
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--team',required=True);a=p.parse_args()
    ok=all(v for _,v in run(Path(a.team)));raise SystemExit(0 if ok else 1)
