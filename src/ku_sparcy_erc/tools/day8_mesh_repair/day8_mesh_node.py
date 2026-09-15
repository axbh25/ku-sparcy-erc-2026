#!/usr/bin/env python3
"""Additive measured-mesh Day8 entry point. Original team nodes are not edited.

Execute authorizes at most one separately audited forward precision-dock, and
only with repair_allow_precision_dock:=true. The entire nominal placement is
screened first, then geometry is checked again at the achieved dock. Original
live support/release/deposit interlocks remain in the inherited state machine.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

# In an installed kit: TEAM/tools/day8_mesh_repair. Explicit source import makes
# a stale colcon overlay unable to select a different version silently.
HERE=Path(__file__).resolve().parent
TEAM=Path(os.environ.get('KU_SPARCY_TEAM','/opt/erc_ws/src/ku_sparcy_erc')).resolve()
sys.path.insert(0,str(TEAM));sys.path.insert(0,str(HERE))
import rclpy
from geometry_msgs.msg import Twist
from ku_sparcy_erc import day8_bin_place as original
from ku_sparcy_erc.day8_geometry import (
    GeometryError,transform,apply,red_mask,colour_mask,masked_cloud,
    validate_book_cloud,
)
from ku_sparcy_erc.urdf_kinematics import rpy_matrix,rotation_z
from ku_sparcy_erc.day567_common import atomic_write_json,vector_odom_to_base
import day8_mesh_core as core

original.DEFAULTS.update(
    repair_allow_precision_dock=False,
    repair_max_dock_m=.22,
    repair_dock_speed_mps=.040,
    repair_dock_accel_mps2=.040,
    repair_dock_timeout_sim_sec=16.0,
    repair_compute_budget_sec=20.0,
    repair_geometry_budget_sec=4.0,
)


def odom_matrix(o):
    return transform(rotation_z(float(o[2])),[float(o[0]),float(o[1]),0.])


def transform_scene(scene,M):
    result=dict(scene)
    for key in ('base_T_bin','base_T_table_top'):
        result[key]=(M@np.asarray(scene[key])).tolist()
    return result


class MeasuredMeshDay8(original.Day8BinPlace):
    def __init__(self):
        self.dock_active=False;self.dock_done=False;self.dock_origin=None
        self.placement_anchor=None;self.dock_trace=[];self.dock_evidence=None
        self.mesh_screen=None;self.mesh_future=None;self.mesh_plan=None
        self.mesh_initial=None;self.post_dock_initial=None
        self.placement_counts_start=None;self.placement_max_drift=0.;self.placement_max_yaw=0.
        self.total_max_displacement=0.;self.mesh_capture_stamp=None
        self.dock_v=0.;self.dock_last_sim=None;self.dock_settle=0;self.dock_settle_stamp=None
        super().__init__()
        if not 0.0<float(self.p['repair_max_dock_m'])<=.22:
            raise ValueError('UNSAFE_DOCK_DISTANCE')
        if not 0.01<=float(self.p['repair_dock_speed_mps'])<=.05:
            raise ValueError('UNSAFE_DOCK_SPEED')
        if not .01<=float(self.p['repair_dock_accel_mps2'])<=.04:
            raise ValueError('UNSAFE_DOCK_ACCELERATION')
        self.assets=core.LocalAssets(self.desc)
        self.model=core.FastModel(self.model)
        self.chain=self.model.chain('base_footprint',f'gripper_{self.arm}_grasping_link')
        self.log('[MESH REPAIR] installed bin/table shapes verified; Day4-7 motion unchanged')

    def log(self,text):
        if 'zero base motion only' in text:
            text=text.replace('zero base motion only','mesh-checked precision dock only if explicitly enabled; placement base stationary')
        super().log(text)

    def keep_safe_hold(self):
        if self.dock_active and not self.done:
            publisher=self.cmd
            try:
                self.cmd=None
                super().keep_safe_hold()
            finally:
                self.cmd=publisher
        else:
            super().keep_safe_hold()

    def stop_dock(self):
        if self.cmd is not None:
            self.cmd.publish(Twist());self.command_counts['base_zero']+=1
        self.dock_v=0.

    def finish(self,ok,reason):
        if getattr(self,'done',False):return
        if getattr(self,'dock_active',False):
            self.stop_dock();self.dock_active=False
        super().finish(ok,reason)

    def safe(self):
        # Explicit phase-specific reference; never rewrite the source Day7 file.
        if not self.fresh(self.joint_stamp,self.p['joint_fresh_sec']):return 'STALE_JOINT_STATES'
        if not self.fresh(self.odom_stamp,.55):return 'STALE_ODOMETRY'
        if not self.fresh(self.scan_stamp,self.p['scan_fresh_sec']):return 'STALE_LIDAR'
        if self.sim+1e-3<float(self.day7['simulation_time_sec']):return 'DAY7_RESULT_FROM_ANOTHER_WORLD_EPOCH'
        if not all(math.isfinite(float(x)) for x in self.odom):return 'NONFINITE_ODOMETRY'
        if not all(n in self.joints and math.isfinite(float(self.joints[n])) for n in self.names):return 'NONFINITE_ARM_JOINTS'
        source=[*self.day7['final_odom_xy'],self.day7['final_yaw_rad']]
        self.total_max_displacement=max(self.total_max_displacement,math.hypot(self.odom[0]-source[0],self.odom[1]-source[1]))
        if self.dock_active:
            relative=np.linalg.inv(odom_matrix(self.dock_origin))@odom_matrix(self.odom)
            forward,cross=relative[0,3],relative[1,3]
            angle=abs(math.atan2(relative[1,0],relative[0,0]))
            if not -.005<=forward<=self.dock_target+.010:return 'DOCK_DISTANCE_ENVELOPE'
            if abs(cross)>.004 or angle>math.radians(.25):return 'DOCK_CROSS_TRACK_OR_HEADING_ERROR'
            if (self.front is not None and self.front.minimum_clearance_m<.55
                    and self.front.hard_cluster_point_count>=3):return 'DOCK_FRONT_LIDAR_HARD_STOP'
            if time.monotonic()-self.dock_started_wall>120:return 'DOCK_WALL_TIMEOUT'
            if self.sim-self.dock_started_sim>self.p['repair_dock_timeout_sim_sec']:return 'DOCK_SIM_TIMEOUT'
            if max(abs(self.joints[n]-self.dock_arm[n]) for n in self.names)>.025:return 'ARM_CHANGED_DURING_DOCK'
        else:
            anchor=self.placement_anchor if self.placement_anchor is not None else source
            drift=math.hypot(self.odom[0]-anchor[0],self.odom[1]-anchor[1])
            dyaw=abs(math.atan2(math.sin(self.odom[2]-anchor[2]),math.cos(self.odom[2]-anchor[2])))
            self.max_drift=max(self.max_drift,drift);self.max_yaw_drift=max(self.max_yaw_drift,dyaw)
            if self.placement_anchor is not None:
                self.placement_max_drift=max(self.placement_max_drift,drift)
                self.placement_max_yaw=max(self.placement_max_yaw,dyaw)
            if drift>.025 or dyaw>math.radians(1.5):return 'PLACEMENT_BASE_MOVED'
        if self.initial_other and max(abs(self.joints[n]-v) for n,v in self.initial_other.items())>.06:
            return 'UNSELECTED_ARM_OR_TORSO_MOVED'
        if not self.fresh(self.last_contacts_rx,.8):return 'ROBOT_CONTACT_STREAM_STALE'
        if self.first_book_bin_time is not None and not self.fresh(self.last_bin_rx,.8):return 'BIN_CONTACT_STREAM_STALE_AFTER_CONTACT'
        if self.release_started is None:
            value=self.joints.get(self.gripper_name)
            if value is None or not math.isfinite(float(value)) or not -1e-6<=float(value)<=self.p['gripper_retention_max_position_m']:
                return 'PRE_RELEASE_GRIPPER_POSITION_INVALID'
        return None

    def clouds(self):
        if self.rgb_stamp is None or not self.depths or not self.fresh(self.rgb_stamp,.55):
            raise GeometryError('RGBD_NOT_FRESH')
        ds,depth=min(self.depths,key=lambda v:abs(v[0]-self.rgb_stamp))
        if abs(ds-self.rgb_stamp)>.2 or not self.fresh(ds,.55):raise GeometryError('RGBD_NOT_SYNCHRONIZED')
        M=self.matrix('base_footprint',self.depth_info.header.frame_id,ds)
        registration=self.matrix(self.rgb_info.header.frame_id,self.depth_info.header.frame_id,ds)
        K,L=original.modelK(self.rgb_info),original.modelK(self.depth_info)
        red,_=masked_cloud(red_mask(self.rgb),depth,K,L,M,step=1,depth_to_rgb=registration)
        predicted=self.tip()@self.Tattach
        local=apply(np.linalg.inv(predicted),red)
        red=red[~(abs(local)<core.BOOK_HALF+.04).all(1)]
        white=(self.rgb.max(2)-self.rgb.min(2)<12)&(self.rgb.min(2)>160)
        wp,_=masked_cloud(white,depth,K,L,M,step=1,depth_to_rgb=registration)
        book,_=masked_cloud(colour_mask(self.rgb,self.colour),depth,K,L,M,step=1,depth_to_rgb=registration)
        observation=validate_book_cloud(book,self.tip(),self.Tattach)
        return red,wp,observation,ds,registration

    def capture(self):
        if self.rgb_stamp is None or self.rgb_stamp==self.mesh_capture_stamp:return False
        self.mesh_capture_stamp=self.rgb_stamp
        red,white,observation,ds,reg=self.clouds()
        initial=self.post_dock_initial
        near=vector_odom_to_base(self.day7['locked_bin_odom_xy'],self.odom[:2],self.odom[2])
        measured=core.measured_scene(red,white,self.assets,near,
            prior_table=None if initial is None else initial['base_T_table_top'],
            initial=initial,budget=float(self.p['repair_geometry_budget_sec']))
        measured.update(rgb_stamp=self.rgb_stamp,depth_stamp=ds,rgb_depth_skew_sec=abs(ds-self.rgb_stamp),
            measured_at_sim=self.sim,rgb_frame=self.rgb_info.header.frame_id,
            depth_frame=self.depth_info.header.frame_id,source_odom=list(self.odom),
            source='live_RGBD_registered_installed_mesh',mesh_sha256=core.BIN_SHA,
            rim_edge_support_provenance='pre_dock_RGBD_plus_post_dock_surface_registration' if initial is not None else 'current_RGBD')
        if self.scenes and np.linalg.norm(np.asarray(self.scenes[-1]['base_T_bin'])[:3,3]-np.asarray(measured['base_T_bin'])[:3,3])>.012:
            self.scenes=[]
        self.scenes.append(measured);self.book_observation=observation
        self.sensor_proof.update(max_rgb_depth_skew_sec=abs(ds-self.rgb_stamp),depth_age_sec=self.sim-ds,
            live_tip_identity_observations=len(self.ledger.tip_samples),depth_to_rgb_transform=reg.tolist())
        if len(self.scenes)>=3:
            self.scene=measured;self.scene['confirming_frames']=len(self.scenes)
            self.evidence_image('bin_geometry');return True
        return False

    def head_survey(self):
        self.finish(False,'MEASURED_BIN_TABLE_VIEW_INCOMPLETE: no automatic head or arm search')

    def scene_still_consistent(self,label):
        red,_,observation,ds,_=self.clouds() if self.release_started is None else self.bin_cloud_after_release()
        B=np.asarray(self.scene['base_T_bin']);local=apply(np.linalg.inv(B),red)
        local=local[(abs(local[:,0])<.34)&(abs(local[:,1])<.21)&(local[:,2]>-.235)&(local[:,2]<.03)]
        if len(local)<80:raise GeometryError('MESH_RECHECK_TOO_FEW_POINTS')
        local=local[::max(1,len(local)//400)]
        _,_,distance=core.closest_triangles(local,self.assets.bin_tri)
        value=float(np.percentile(distance,80))
        if value>.008:raise GeometryError('BIN_MESH_MOVED_OR_RECHECK_INCONSISTENT')
        self.scene_live_checks.append(dict(stage=label,sim_time=self.sim,rgb_stamp=self.rgb_stamp,
            depth_stamp=ds,points=len(local),surface_residual80_m=value,passed=True,actual_mesh=True))

    def bin_cloud_after_release(self):
        # The release/retract states must not require a now-intentionally absent
        # held-book cloud. They retain contact/support/deposit interlocks.
        ds,depth=min(self.depths,key=lambda v:abs(v[0]-self.rgb_stamp))
        if not self.fresh(ds,.55) or not self.fresh(self.rgb_stamp,.55) or abs(ds-self.rgb_stamp)>.2:
            raise GeometryError('POST_RELEASE_RGBD_STALE')
        M=self.matrix('base_footprint',self.depth_info.header.frame_id,ds)
        reg=self.matrix(self.rgb_info.header.frame_id,self.depth_info.header.frame_id,ds)
        red,_=masked_cloud(red_mask(self.rgb),depth,original.modelK(self.rgb_info),original.modelK(self.depth_info),M,depth_to_rgb=reg)
        return red,None,None,ds,None

    def settings(self):
        p=dict(self.p);p['book_margin_m']=max(float(p['book_margin_m']),float(self.book_observation['uncertainty_margin_m']))
        p['joint_sample_rad']=.025
        return p

    def begin_plan(self):
        restore=self.day7.get('compact_carry_target_positions_rad')
        if not isinstance(restore,list) or len(restore)!=7:raise GeometryError('KNOWN_COMPACT_CARRY_REQUIRED')
        self.snapshot_joints=dict(self.joints);self.mesh_initial=dict(self.scene)
        self.mesh_screen=core.MeshScreen(self.urdf_path,self.model,self.arm,original.share,
            self.assets,dict(self.joints),self.Tattach,self.settings())
        self.envelopes=self.mesh_screen
        rotation=rpy_matrix(self.day5['grasp_plan']['candidate_plans'][self.arm]['target_rpy_base'])
        self.mesh_rotation=rotation
        offsets=(0.,) if self.dock_done else tuple(x for x in (0.,.18,.20,.22) if x<=self.p['repair_max_dock_m'])
        deadline=time.monotonic()+float(self.p['repair_compute_budget_sec'])
        self.planning_started_wall=time.monotonic()
        self.future=self.pool.submit(core.build_complete_plan,self.mesh_screen,dict(self.joints),restore,
            rotation,dict(self.scene),self.settings(),deadline,offsets)
        self.transition('MESH_PLAN_COMPLETE','screen dock sweep, pre-place, lowering, opening envelope and nominal retract before moving')

    def begin_lower_plan(self):
        self.snapshot_joints=dict(self.joints);self.planning_started_wall=time.monotonic()
        args=(self.mesh_screen,dict(self.joints),self.plan,self.mesh_rotation,dict(self.scene),self.settings())
        def calculate():
            start=time.monotonic();seg=core.plan_lower(*args,start+float(self.p['repair_compute_budget_sec']))
            return dict(passed=True,segment=seg,planning_wall_sec=time.monotonic()-start,new_ik_from_actual_preplace_joints=True)
        self.future=self.pool.submit(calculate)
        self.transition('PLAN_LOWER_FAST','revalidate the measured-mesh lowering from actual settled joints')

    def screen_actual_retract(self,joints,actual):
        return core.plan_retract(self.mesh_screen,dict(joints),self.plan,self.mesh_rotation,
            dict(self.scene),self.settings(),np.asarray(self.release_geometry),
            time.monotonic()+float(self.p['repair_compute_budget_sec']))

    def _tick(self):
        if self.state not in ('MESH_PLAN_COMPLETE','MESH_DOCK','MESH_DOCK_SETTLE','MESH_DOCK_REFRESH'):
            return super()._tick()
        if time.monotonic()-self.last_clock_wall>self.p['clock_stall_wall_timeout_sec']:
            self.finish(False,'CLOCK_STALL_WALL_WATCHDOG');return

        # After the precision dock, allow ROS subscription callbacks to catch up
        # before re-entering the inherited Day8 freshness interlocks.  We do not
        # manufacture timestamps or relax any freshness threshold: progression
        # requires genuinely fresh joint-state, odometry and LiDAR messages.
        if self.state=='MESH_DOCK_REFRESH':
            self.keep_safe_hold()
            if time.monotonic()-self.post_dock_refresh_started_wall>2.0:
                self.finish(False,'POST_DOCK_SENSOR_REFRESH_TIMEOUT');return
            if (self.fresh(self.joint_stamp,self.p['joint_fresh_sec'])
                    and self.fresh(self.odom_stamp,.55)
                    and self.fresh(self.scan_stamp,self.p['scan_fresh_sec'])):
                self.transition('MEASURE_BIN',
                    'fresh post-dock joint/odom/LiDAR callbacks received; reacquire book/bin/table registration')
            return

        reason=self.safe()
        if reason:self.finish(False,reason);return
        self.keep_safe_hold()
        if self.state=='MESH_PLAN_COMPLETE':
            if not self.future.done():
                if time.monotonic()-self.planning_started_wall>float(self.p['repair_compute_budget_sec'])+.5:
                    self.finish(False,'MESH_PLANNING_WATCHDOG')
                return
            self.mesh_plan=self.future.result()
            if max(abs(self.joints[n]-self.snapshot_joints[n]) for n in self.names)>.015:
                self.finish(False,'ARM_MOVED_DURING_MESH_PLANNING');return
            self.plan=self.mesh_plan
            self.log('[FULL MESH PLAN][PASS] forward_dock_m='+str(self.plan['docking_forward_m'])+
                     ' computation_sec='+str(self.plan['planning_wall_sec']))
            if self.op=='plan':self.finish(True,'preview: hypothetical dock plus complete nominal arm path; no actuator commands');return
            distance=float(self.plan['docking_forward_m'])
            if distance>.003:
                if not self.p['repair_allow_precision_dock']:
                    self.finish(False,'PRECISION_DOCK_REQUIRED_BUT_NOT_AUTHORIZED');return
                if self.cmd.get_subscription_count()<1 or self.count_publishers('/cmd_vel')!=1:
                    self.finish(False,'BASE_COMMAND_OWNERSHIP_NOT_EXCLUSIVE');return
                # Mandatory fresh retained-book and bin check immediately before dock.
                self.scene_still_consistent('pre_dock')
                self.dock_origin=list(self.odom);self.dock_target=distance
                self.dock_arm={n:self.joints[n] for n in self.names}
                self.dock_started_sim=self.sim;self.dock_started_wall=time.monotonic()
                self.dock_last_sim=self.sim;self.dock_active=True;self.dock_book_checks=[]
                self.transition('MESH_DOCK','short separate forward precision dock; arm fixed; no lateral/yaw command')
                return
            self.scene_still_consistent('pre_place')
            if self.placement_anchor is None:self.placement_anchor=list(self.odom)
            if self.placement_counts_start is None:self.placement_counts_start=dict(self.command_counts)
            self.evidence_image('plan');self.request('PRE_PLACE');return
        relative=np.linalg.inv(odom_matrix(self.dock_origin))@odom_matrix(self.odom)
        x=float(relative[0,3]);remaining=self.dock_target-x
        if self.state=='MESH_DOCK':
            dt=max(0.,min(.20,self.sim-self.dock_last_sim));self.dock_last_sim=self.sim
            accel=float(self.p['repair_dock_accel_mps2'])
            desired=min(float(self.p['repair_dock_speed_mps']),.7*max(0.,remaining),
                        math.sqrt(2*accel*max(0.,remaining-.002)))
            self.dock_v+=float(np.clip(desired-self.dock_v,-accel*dt,accel*dt))
            if remaining<=.003 and self.dock_v<=.003:
                self.stop_dock();self.dock_settle=0;self.dock_settle_stamp=None
                self.dock_last_x=x;self.transition('MESH_DOCK_SETTLE','zero base command; verify achieved pose before remeasurement');return
            if time.monotonic()-getattr(self,'last_dock_book_check_wall',0)>1.:
                _,_,held,ds,_=self.clouds()
                self.dock_book_checks.append(dict(sim=self.sim,rgb_stamp=self.rgb_stamp,depth_stamp=ds,**held))
                self.last_dock_book_check_wall=time.monotonic()
            msg=Twist();msg.linear.x=self.dock_v;self.cmd.publish(msg)
            self.command_counts['base_nonzero' if self.dock_v>1e-8 else 'base_zero']+=1
            self.dock_trace.append(dict(sim=self.sim,x_m=x,cross_m=float(relative[1,3]),
                yaw_rad=math.atan2(relative[1,0],relative[0,0]),command_vx_mps=self.dock_v,
                lidar_clearance_m=float(self.front.minimum_clearance_m),scan_stamp=self.scan_stamp,
                lidar_cluster_points=int(self.front.hard_cluster_point_count),
                arm_max_delta_rad=max(abs(self.joints[n]-self.dock_arm[n]) for n in self.names),
                held_contact_last=self.ledger.tip_last))
            if len(self.dock_trace)>2500:self.dock_trace=self.dock_trace[-2500:]
            return
        self.stop_dock()
        if self.odom_stamp!=self.dock_settle_stamp:
            self.dock_settle=self.dock_settle+1 if abs(x-self.dock_last_x)<.001 else 0
            self.dock_last_x=x;self.dock_settle_stamp=self.odom_stamp
        if self.dock_settle>=4:
            achieved=list(self.odom);delta=np.linalg.inv(odom_matrix(achieved))@odom_matrix(self.dock_origin)
            self.post_dock_initial=transform_scene(self.mesh_initial,delta)
            self.dock_evidence=dict(passed=True,source_odom=self.dock_origin,achieved_odom=achieved,
                target_forward_m=self.dock_target,actual_forward_m=x,source_scene=self.mesh_initial,
                planned_sweep_samples_m=self.mesh_plan['dock_mesh_sweep_samples_m'],trace=self.dock_trace,
                arm_was_stationary=True,held_book_checks=self.dock_book_checks,completed_sim=self.sim)
            self.dock_active=False;self.dock_done=True;self.placement_anchor=achieved
            self.placement_counts_start=dict(self.command_counts)
            self.max_drift=0.;self.max_yaw_drift=0.
            self.scenes=[];self.mesh_capture_stamp=None;self.processed_rgb=None
            self.post_dock_refresh_started_wall=time.monotonic()
            self.transition('MESH_DOCK_REFRESH',
                'dock complete; waiting for genuinely fresh joint/odom/LiDAR callbacks before remeasurement')

    def write(self):
        super().write()
        data=json.loads(Path(self.result_path).read_text())
        counts=getattr(self,'placement_counts_start',None)
        phase_counts=None if counts is None else {k:self.command_counts.get(k,0)-counts.get(k,0) for k in self.command_counts}
        data['base_motion_metric_scope']='stationary_placement_after_separately_audited_precision_dock'
        data['mesh_repair']=dict(version=core.VERSION,docking=self.dock_evidence,
            docking_authorized=bool(self.p.get('repair_allow_precision_dock',False)),
            placement_anchor_odom=self.placement_anchor,placement_phase_command_counts=phase_counts,
            placement_max_base_drift_m=self.placement_max_drift,
            placement_max_base_yaw_drift_deg=math.degrees(self.placement_max_yaw),
            total_displacement_from_original_day7_m=self.total_max_displacement,
            docking_and_placement_are_separately_audited=True)
        atomic_write_json(self.result_path,data)


def main():
    from verify_compatibility import verify
    verify(TEAM)
    rclpy.init();node=None;code=1
    try:
        node=MeasuredMeshDay8()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node,timeout_sec=.05)
            if node.op!='plan':node.keep_safe_hold()
        code=0 if node.passed else 1
    except KeyboardInterrupt:
        if node:node.finish(False,'OPERATOR_INTERRUPT')
    except Exception as exc:
        if node:node.finish(False,type(exc).__name__+': '+str(exc))
        else:print('[MESH REPAIR][FAIL]',type(exc).__name__,str(exc),flush=True)
    finally:
        if node:
            node.dock_active=False;node.stop_dock()
            until=time.monotonic()+.4
            while rclpy.ok() and time.monotonic()<until:
                node.keep_safe_hold();rclpy.spin_once(node,timeout_sec=.025)
            node.pool.shutdown(wait=False,cancel_futures=True);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    raise SystemExit(code)

if __name__=='__main__':main()
