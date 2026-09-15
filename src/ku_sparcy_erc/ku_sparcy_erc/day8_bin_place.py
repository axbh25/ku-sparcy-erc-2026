#!/usr/bin/env python3
"""Day 8: base-stationary placement, phase-aware support/release verification.\n\nDAY8_CLEAR_VIEW_TRANSPORT_READY_V1

plan: no actuator publishers; live sensor measurement plus all-path IK/screen.
survey: base zero + position hold + bounded head views, no manipulation.
execute: remeasure/replan, then gated or autonomous arm-only deposit.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as SENSOR
from rclpy.time import Time
from rclpy.clock import Clock as RclClock, ClockType
from rclpy.duration import Duration
from ament_index_python.packages import get_package_share_directory as share
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image,CameraInfo,JointState,LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from std_msgs.msg import String
from ros_gz_interfaces.msg import Contacts
import tf2_ros
from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel,rpy_matrix,rotation_vector
from ku_sparcy_erc.range_fusion import lidar_corridor_clearance,normalize_depth_array
from ku_sparcy_erc.day567_common import vector_odom_to_base,yaw_from_odom,atomic_write_json
from ku_sparcy_erc.day8_geometry import (GeometryError,transform,apply,quat_matrix,box_corners,
    official_bin_profile,red_mask,colour_mask,masked_cloud,fit_bin,book_attachment,
    validate_book_cloud,supported_geometry,book_bounds_in_bin)
from ku_sparcy_erc.day8_contacts import ContactLedger,names_of,is_bin
from ku_sparcy_erc.day8_planner import (
    CollisionGeometry,
    build_placement_plan,
    plan_actual_retract,
    build_clear_view_plan,
    build_fast_placement_plan,
    build_fast_preplace_plan,
    build_fast_lower_plan,
)

DEFAULTS=dict(operation='plan',manual_approval_required=True,
 day7_result_path='/opt/erc_ws/src/ku_sparcy_erc/day7_result.json',
 result_path='/opt/erc_ws/src/ku_sparcy_erc/day8_result.json',
 image_output_dir='/opt/erc_ws/src/ku_sparcy_erc/erc_images',
 ready_wall_timeout_sec=60.,clock_stall_wall_timeout_sec=20.,approval_wall_timeout_sec=600.,
 planning_wall_timeout_sec=6.,fast_plan_budget_sec=3.0,fast_plan_joint_samples=8,fast_plan_ik_iterations=90,fast_preplace_budget_sec=2.5,fast_lower_budget_sec=1.5,fast_stage_ik_iterations=45,fast_stage_joint_samples=7,fast_stage_seed_retries=0,contact_stream_fresh_sec=.8,
 joint_fresh_sec=.55,image_fresh_sec=.55,scan_fresh_sec=.60,max_rgb_depth_skew_sec=.20,
 tf_latest_fallback_max_sec=.05,
 geometry_confirm_frames=3,geometry_timeout_sec=6.,base_drift_tolerance_m=.025,
 base_yaw_tolerance_deg=1.5,retention_contact_stale_sec=.8,
 head_survey_pan_offsets_rad=[0.,.35,-.35,.55,-.55,0.],
 head_survey_tilts_rad=[-.15,-.25,-.25,-.40,-.40,-.55],head_motion_sec=2.5,head_timeout_sec=12.,
 head_tolerance_rad=.04,joint_tolerance_rad=.03,settle_samples=4,stage_timeout_margin_sec=4.,
 bin_pose_drift_tolerance_m=.025,front_hard_stop_m=.55,lidar_corridor_half_width_m=.42,
 gripper_hold_position_m=.01,gripper_hold_period_sec=.20,
 gripper_retention_max_position_m=.020,
 gripper_release_positions_m=[.018,.026,.034,.04],gripper_release_step_sec=.80,
 gripper_open_tolerance_m=.007,support_confirm_sec=.4,support_fresh_sec=.45,placement_max_penetration_m=.006,
 release_verify_sec=.8,deposit_verify_sec=2.,deposit_timeout_sec=7.,fingertip_quiet_sec=.8,
 book_margin_m=.012,robot_margin_m=.008,rim_clearance_m=.055,cart_step_m=.012,
 lower_speed_mps=.018,preplace_speed_mps=.050,retract_speed_mps=.045,
 max_joint_speed_radps=.16,retract_away_m=.10,
 clear_view_back_m=.04,clear_view_lateral_m=.12,clear_view_up_m=.05,
 clear_view_min_cross_track_m=.16,clear_view_speed_mps=.035,
 clear_view_max_joint_delta_rad=.45)


def load(path):
    with open(path,encoding='utf-8') as f:return json.load(f)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(msg):return float(msg.header.stamp.sec)+msg.header.stamp.nanosec*1e-9


def modelK(msg):
    if msg.k[0]<=0 or msg.k[4]<=0 or not msg.header.frame_id:
        raise GeometryError('INVALID_CAMERAINFO')
    return [float(msg.k[i]) for i in (0,4,2,5)]


class Day8BinPlace(Node):
    RGB='/head_front_camera/head_front_camera/color/image_raw'
    DEPTH='/head_front_camera/head_front_camera/depth/image_rect_raw'
    RGB_INFO='/head_front_camera/head_front_camera/color/camera_info'
    DEPTH_INFO='/head_front_camera/head_front_camera/depth/camera_info'

    def __init__(self):
        super().__init__('ku_sparcy_day8_bin_place')
        for k,v in DEFAULTS.items():self.declare_parameter(k,v)
        self.p={k:self.get_parameter(k).value for k in DEFAULTS}
        self.op=self.p['operation'];self.result_path=self.p['result_path'];self.manual=self.p['manual_approval_required']
        if self.op not in ('survey','plan','execute'):raise ValueError('operation must be survey, plan, or execute')
        if self.p['max_rgb_depth_skew_sec']>.20 or self.p['image_fresh_sec']>.55:
            raise ValueError('RGB/depth safety limits may not be loosened')
        if self.p['lower_speed_mps']>.025:raise ValueError('Day 8 lowering must be <=0.025 m/s')
        for key,lo,hi in (('book_margin_m',.010,.030),('robot_margin_m',.005,.025),
                          ('rim_clearance_m',.040,.100),('cart_step_m',.004,.020),
                          ('preplace_speed_mps',.020,.080),('retract_speed_mps',.020,.070),
                          ('max_joint_speed_radps',.050,.200)):
            if not lo<=float(self.p[key])<=hi:raise ValueError('UNSAFE_PARAMETER:'+key)
        if not .02<=float(self.p['clear_view_back_m'])<=.10:
            raise ValueError('UNSAFE_PARAMETER:clear_view_back_m')
        if not .08<=float(self.p['clear_view_lateral_m'])<=.22:
            raise ValueError('UNSAFE_PARAMETER:clear_view_lateral_m')
        if not .03<=float(self.p['clear_view_up_m'])<=.12:
            raise ValueError('UNSAFE_PARAMETER:clear_view_up_m')
        if not .020<=float(self.p['clear_view_speed_mps'])<=.060:
            raise ValueError('UNSAFE_PARAMETER:clear_view_speed_mps')
        if not .20<=float(self.p['clear_view_max_joint_delta_rad'])<=.70:
            raise ValueError('UNSAFE_PARAMETER:clear_view_max_joint_delta_rad')
        if len(self.p['head_survey_pan_offsets_rad']) != len(self.p['head_survey_tilts_rad']):
            raise ValueError('HEAD_SURVEY_PAN_TILT_LENGTH_MISMATCH')
        release=self.p['gripper_release_positions_m']
        if (not release or release!=sorted(release) or min(release)<=.01 or max(release)>.069):
            raise ValueError('invalid public gripper release sequence')
        self.day7=load(self.p['day7_result_path'])
        if self.day7.get('passed') is not True or self.day7.get('day')!=7:raise ValueError('Day 7 input is not a PASS')
        # Placement consumes a generic live transport handoff.  It must not
        # depend on the exact retreat/rotation/transit state machine that
        # produced the endpoint.
        if self.day7.get('bin_detection_uses_live_rgb_depth_tf') is not True:
            raise ValueError('Day 7 missing live bin perception proof')
        for k in ('final_odom_xy','final_yaw_rad','locked_bin_odom_xy'):
            if self.day7.get(k) is None:
                raise ValueError('Day 7 missing placement handoff field '+k)
        if self.day7.get('unintended_robot_contacts')!=0 or self.day7.get('bin_contact_messages')!=0:
            raise ValueError('Day 7 input contains unintended/premature contacts')
        self.day5=load(self.day7['day5_result_path']);self.day4=load(self.day5['source_result_path'])
        if self.day5.get('passed') is not True or self.day4.get('passed') is not True:
            raise ValueError('Day 4/5 input is not a passed result')
        self.arm=self.day7['selected_arm']
        if self.arm not in ('left','right') or self.day5.get('selected_arm')!=self.arm or not self.day5.get('retention_verified'):
            raise ValueError('Day 5/Day 7 selected arm or retention disagrees')
        self.colour=self.day5['requested_book_colour']
        self.names=[f'arm_{self.arm}_{i}_joint' for i in range(1,8)]
        self.other_names=[f'arm_{"right" if self.arm=="left" else "left"}_{i}_joint' for i in range(1,8)]
        self.gripper_name=f'gripper_{self.arm}_finger_joint'
        self.source_digest=sha(self.p['day7_result_path'])
        self.desc=share('erc_description');self.urdf_path=os.path.join(self.desc,'urdf','tiago_pro.urdf')
        self.model=URDFKinematicModel.from_file(self.urdf_path)
        self.chain=self.model.chain('base_footprint',f'gripper_{self.arm}_grasping_link')
        self.mesh_path=os.path.join(self.desc,'models','collection_bin','meshes','erc_base_collection_bin.STL')
        self.profile=official_bin_profile(self.mesh_path)
        self.urdf_digest=sha(self.urdf_path)
        self._verify_interfaces()
        self.done=False;self.passed=False;self.state='WAIT_INPUTS';self.reason=''
        self.sim=None;self.started_sim=None;self.last_clock_wall=time.monotonic();self.wall_start=time.monotonic()
        self.phase_start=None;self.approval_start_wall=None;self.pending=None
        self.joints={};self.joint_stamp=None;self.odom=None;self.odom_stamp=None
        self.front=None;self.scan_stamp=None;self.ledger=ContactLedger(self.arm)
        self.last_contacts_rx=None;self.last_bin_rx=None
        self.bridge=CvBridge();self.rgb=None;self.rgb_stamp=None;self.rgb_info=None;self.depth_info=None;self.depths=[]
        self.processed_rgb=None;self.precheck_rgb_stamp=None
        self.capture_attempts=[];self.scenes=[];self.scene=None;self.plan=None
        self.clear_view_plan=None;self.clear_view_snapshot=None;self.clear_view_done=False
        self.Tattach=None;self.envelopes=None;self.book_observation=None
        self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
        self.snapshot_joints=None;self.initial_other=None;self.max_drift=0.;self.max_yaw_drift=0.
        self.command_counts=dict(base_zero=0,base_nonzero=0,arm=0,head=0,gripper=0)
        self.commanded_gripper=float(self.p['gripper_hold_position_m']);self.last_grip_publish=None
        self.lower_contact_seen=False;self.support_verified=False;self.release_started=None
        self.release_complete=None;self.release_verified=False;self.retract_done=None;self.deposit_verified=False
        self.first_book_bin_time=None;self.closed_position_samples=0
        self.completed=[];self.history=[];self.evidence={};self.evidence_stamps={};self.release_trace=[]
        self.motion_name=None;self.motion_q=None;self.motion_started=None;self.motion_deadline=None
        self.settle_count=0;self.settle_stamp=None;self.lower_index=0;self.release_index=0
        self.release_step_start=None;self.head_index=0;self.head_target=None
        self.visual_deposit=[];self.sensor_proof={};self.release_geometry=None
        self.last_status_wall=0.;self.last_audit_sim=None;self.planning_started_wall=None
        self.retract_rescreen=None;self.scene_live_checks=[];self.tf_latest_fallbacks=0
        self.max_book_bin_penetration=0.;self.book_bin_depth_samples=0;self.abort_positions=None
        self.tf=tf2_ros.Buffer(cache_time=Duration(seconds=15.));self.tfl=tf2_ros.TransformListener(self.tf,self)
        self.create_subscription(Clock,'/clock',self.clock_cb,SENSOR)
        self.create_subscription(JointState,'/joint_states',self.joint_cb,20)
        self.create_subscription(Odometry,'/odom',self.odom_cb,SENSOR)
        self.create_subscription(LaserScan,'/scan_front_raw',self.scan_cb,SENSOR)
        self.create_subscription(Contacts,'/contacts',lambda m:self.contacts_cb(m,False),SENSOR)
        self.create_subscription(Contacts,'/bin_contacts',lambda m:self.contacts_cb(m,True),SENSOR)
        self.create_subscription(CameraInfo,self.RGB_INFO,lambda m:setattr(self,'rgb_info',m),SENSOR)
        self.create_subscription(CameraInfo,self.DEPTH_INFO,lambda m:setattr(self,'depth_info',m),SENSOR)
        self.create_subscription(Image,self.RGB,self.rgb_cb,SENSOR)
        self.create_subscription(Image,self.DEPTH,self.depth_cb,SENSOR)
        self.create_subscription(String,'/ku_sparcy/day8/approve',self.approve,10)
        self.status=self.create_publisher(String,'/ku_sparcy/day8/status',10)
        self.cmd=None;self.arm_pub=None;self.grip_pub=None;self.head_pub=None
        if self.op!='plan':
            self.cmd=self.create_publisher(Twist,'/cmd_vel',10)
            self.grip_pub=self.create_publisher(JointTrajectory,f'/gripper_{self.arm}_controller/joint_trajectory',10)
            self.head_pub=self.create_publisher(JointTrajectory,'/head_controller/joint_trajectory',10)
        if self.op=='execute':
            self.arm_pub=self.create_publisher(JointTrajectory,f'/arm_{self.arm}_controller/joint_trajectory',10)
        # A steady-clock timer can stop a command even if simulated /clock stalls.
        self.timer=self.create_timer(.05,self.tick,clock=RclClock(clock_type=ClockType.STEADY_TIME))
        self.log('[DAY8] '+self.op+'; selected arm='+self.arm+'; zero base motion only')

    def log(self,text):self.get_logger().info(text)

    def _verify_interfaces(self):
        book_file=Path(self.desc)/'models/book/sdf/erc_book.sdf'
        sizes=[list(map(float,e.text.split())) for e in ET.parse(book_file).getroot().findall('.//geometry/box/size')]
        if not sizes or any(not np.allclose(v,[.25,.02,.16],atol=1e-8) for v in sizes):
            raise ValueError('INSTALLED_BOOK_SHAPE_NOT_REVIEWED_2CM_GEOMETRY')
        r=ET.parse(self.urdf_path).getroot()
        found=False
        for j in r.findall('.//ros2_control/joint'):
            if j.get('name')==self.gripper_name:
                modes=[e.get('name') for e in j.findall('command_interface')]
                if modes!=['position']:raise ValueError('GRIPPER_NOT_POSITION_ONLY')
                found=True
        if not found:raise ValueError('PUBLIC_GRIPPER_JOINT_MISSING')
        for tilt in self.p['head_survey_tilts_rad']:
            lim=self.model.joints_by_name['head_2_joint']
            if not lim.lower+.02<float(tilt)<lim.upper-.02:
                raise ValueError('HEAD_HARD_LIMIT_MARGIN')
        for offset in self.p['head_survey_pan_offsets_rad']:
            if abs(float(offset))>.60:
                raise ValueError('HEAD_SURVEY_PAN_OFFSET_TOO_LARGE')

    def clock_cb(self,m):
        t=float(m.clock.sec)+m.clock.nanosec*1e-9
        if self.sim is not None and t<self.sim-1e-6:
            self.finish(False,'SIMULATION_CLOCK_RESET');return
        if self.sim is None or t>self.sim:self.last_clock_wall=time.monotonic()
        self.sim=t
        if self.started_sim is None:self.started_sim=t

    def joint_cb(self,m):
        self.joints.update({n:float(v) for n,v in zip(m.name,m.position)})
        self.joint_stamp=stamp(m) or self.sim
        # Effort is deliberately neither commanded nor used by Day 8.

    def odom_cb(self,m):
        self.odom=(float(m.pose.pose.position.x),float(m.pose.pose.position.y),float(yaw_from_odom(m)))
        self.odom_stamp=stamp(m) or self.sim

    def matrix(self,target,source,t=None):
        when=Time() if t is None else Time(seconds=float(t))
        try:
            tr=self.tf.lookup_transform(target,source,when)
        except Exception:
            if t is None:
                raise
            latest=self.tf.lookup_transform(target,source,Time())
            actual=stamp(latest)
            if actual and abs(float(t)-actual)>float(self.p['tf_latest_fallback_max_sec']):
                raise
            tr=latest
            self.tf_latest_fallbacks+=1
        q=tr.transform.rotation;v=tr.transform.translation
        return transform(quat_matrix([q.x,q.y,q.z,q.w]),[v.x,v.y,v.z])

    def scan_cb(self,m):
        try:
            tr=self.tf.lookup_transform('base_footprint',m.header.frame_id,Time())
            t=tr.transform.translation;q=tr.transform.rotation
            self.front=lidar_corridor_clearance(m.ranges,m.angle_min,m.angle_increment,m.range_min,m.range_max,
                [t.x,t.y,t.z],[q.x,q.y,q.z,q.w],self.p['lidar_corridor_half_width_m'],10.,self.p['front_hard_stop_m'])
            self.scan_stamp=stamp(m) or self.sim
        except Exception:pass

    def rgb_cb(self,m):
        try:self.rgb=self.bridge.imgmsg_to_cv2(m,'bgr8');self.rgb_stamp=stamp(m) or self.sim
        except Exception:pass

    def depth_cb(self,m):
        try:
            arr=normalize_depth_array(self.bridge.imgmsg_to_cv2(m,'passthrough'),m.encoding)
            self.depths.append((stamp(m) or self.sim,arr));self.depths=self.depths[-35:]
        except Exception:pass

    def contacts_cb(self,m,bin_stream):
        if self.sim is None or self.done:return
        t=stamp(m) if hasattr(m,'header') else self.sim
        t=t or self.sim
        if t>self.sim+.03 or self.sim-t>.8:return
        if bin_stream:self.last_bin_rx=self.sim
        else:self.last_contacts_rx=self.sim
        allowed=self.op=='execute' and self.state in ('LOWERING','SETTLE_SUPPORT','VERIFY_SUPPORT',
            'WAIT_APPROVAL_RELEASE','OPEN_STEPWISE','VERIFY_RELEASE','WAIT_APPROVAL_RETRACT',
            'REPLAN_RETRACT','MOVE_RETRACT','VERIFY_DEPOSIT')
        for c in m.contacts:
            pair=names_of(c)
            reason=self.ledger.feed(pair,t,self.state,allowed)
            if reason:self.finish(False,reason);return
            if self.ledger.held_id in pair and any(is_bin(n) for n in pair):
                depths=[float(d) for d in getattr(c,'depths',[]) if np.isfinite(d)]
                if depths:
                    self.max_book_bin_penetration=max(self.max_book_bin_penetration,max(depths))
                    self.book_bin_depth_samples+=len(depths)
                    if max(depths)>self.p['placement_max_penetration_m']:
                        self.finish(False,'EXCESSIVE_BOOK_BIN_PENETRATION');return
                if self.first_book_bin_time is None:self.first_book_bin_time=t
                if self.state=='LOWERING':
                    if not supported_geometry(self.tip(),self.Tattach,self.scene):
                        self.finish(False,'BOOK_CONTACT_ON_RIM_OR_OUTSIDE_FLOOR_ENVELOPE');return
                    self.lower_contact_seen=True
                    self.halt_arm()
                    self.transition('SETTLE_SUPPORT','book/bin contact detected near planned interior floor')
                    self.motion_q=np.array([self.joints[n] for n in self.names]);self.settle_count=0

    def tip(self,q=None):return self.model.forward(self.chain,self.joints if q is None else q).tip_transform

    def fresh(self,t,age):return self.sim is not None and t is not None and -.03<=self.sim-t<=age

    def ready(self):
        needed=self.names+self.other_names+['torso_lift_joint',self.gripper_name,
                f'gripper_{"right" if self.arm=="left" else "left"}_finger_joint','head_1_joint','head_2_joint']
        return (all(n in self.joints for n in needed) and self.odom is not None
                and self.rgb_info is not None and self.depth_info is not None and self.depths
                and self.rgb is not None and self.front is not None
                and self.ledger.held_id is not None and len(self.ledger.tip_samples)>=3
                and all(np.isfinite(self.joints[n]) for n in needed)
                and all(pub is None or pub.get_subscription_count()>0
                        for pub in (self.cmd,self.grip_pub,self.head_pub,self.arm_pub)))

    def safe(self):
        if not self.fresh(self.joint_stamp,self.p['joint_fresh_sec']):return 'STALE_JOINT_STATES'
        if not self.fresh(self.odom_stamp,.55):return 'STALE_ODOMETRY'
        if not self.fresh(self.scan_stamp,self.p['scan_fresh_sec']):return 'STALE_LIDAR'
        if self.front and self.front.minimum_clearance_m<self.p['front_hard_stop_m'] and self.front.hard_cluster_point_count>=3:
            # The base is commanded zero throughout Day 8.  A nearby bin is
            # therefore not a base-motion emergency; arm/object geometry is
            # screened separately and live contacts remain fatal.
            self.sensor_proof['front_lidar_hard_stop_observed']=True
        sx,sy=self.day7['final_odom_xy'];yaw=self.day7['final_yaw_rad']
        drift=math.hypot(self.odom[0]-sx,self.odom[1]-sy)
        dyaw=abs(math.atan2(math.sin(self.odom[2]-yaw),math.cos(self.odom[2]-yaw)))
        self.max_drift=max(self.max_drift,drift);self.max_yaw_drift=max(self.max_yaw_drift,dyaw)
        if drift>self.p['base_drift_tolerance_m'] or math.degrees(dyaw)>self.p['base_yaw_tolerance_deg']:
            return 'SOURCE_POSE_MOVED'
        if self.sim+1e-3<float(self.day7['simulation_time_sec']):return 'DAY7_RESULT_FROM_ANOTHER_WORLD_EPOCH'
        if self.initial_other:
            if max(abs(self.joints[n]-v) for n,v in self.initial_other.items())>.06:return 'UNSELECTED_ARM_OR_TORSO_MOVED'
        if not self.fresh(self.last_contacts_rx,self.p['contact_stream_fresh_sec']):
            return 'ROBOT_CONTACT_STREAM_STALE'
        # /bin_contacts is event-driven: before the retained book first
        # touches the bin, silence means there is currently no bin contact.
        # Once placement contact has actually begun, however, continuing
        # support evidence must remain fresh.
        if (self.first_book_bin_time is not None
                and not self.fresh(
                    self.last_bin_rx,
                    self.p['contact_stream_fresh_sec'])):
            return 'BIN_CONTACT_STREAM_STALE_AFTER_CONTACT'
        # A stable Gazebo grasp does not necessarily emit a fresh fingertip
        # contact message continuously.  Contact freshness is therefore
        # evidence, not by itself proof that the retained book was lost.
        #
        # Before intentional release, require the physical gripper joint to
        # remain within a conservative closed/retaining envelope.  The
        # independent RGB-D held-book validation performed by MEASURE_BIN
        # remains mandatory before survey/plan acceptance.
        if self.release_started is None:
            actual_gripper = self.joints.get(self.gripper_name)
            if actual_gripper is None:
                return 'PRE_RELEASE_GRIPPER_STATE_MISSING'
            if float(actual_gripper) > float(
                    self.p['gripper_retention_max_position_m']):
                return 'PRE_RELEASE_GRIPPER_OPENED'
        return None

    def trajectory(self,names,points):
        msg=JointTrajectory();msg.joint_names=list(names)
        for positions,t in points:
            p=JointTrajectoryPoint();p.positions=[float(v) for v in positions]
            whole=int(t);p.time_from_start.sec=whole;p.time_from_start.nanosec=int((t-whole)*1e9)
            msg.points.append(p)
        return msg

    def keep_safe_hold(self):
        # Before intentional release, Day 8 owns the retained-grasp target.
        # Never inherit the previous Day-7 0.010 m command accidentally.
        if self.release_started is None:
            self.commanded_gripper = float(
                self.p['gripper_hold_position_m'])

        if self.cmd is not None:
            self.cmd.publish(Twist());self.command_counts['base_zero']+=1

        if (self.grip_pub is not None and self.sim is not None
                and not (self.state=='OPEN_STEPWISE' and self.release_step_start is not None
                         and self.sim-self.release_step_start<self.p['gripper_release_step_sec'])):
            if self.last_grip_publish is None or self.sim-self.last_grip_publish>=self.p['gripper_hold_period_sec']:
                self.grip_pub.publish(
                    self.trajectory(
                        [self.gripper_name],
                        [([self.commanded_gripper], .2)]))
                self.command_counts['gripper']+=1
                self.last_grip_publish=self.sim

    def halt_arm(self):
        if self.arm_pub and all(n in self.joints for n in self.names):
            self.arm_pub.publish(self.trajectory(self.names,[([self.joints[n] for n in self.names],.25)]))
            self.command_counts['arm']+=1

    def transition(self,state,reason):
        self.state=state;self.reason=reason;self.phase_start=self.sim
        self.history.append(dict(state=state,sim_time=self.sim,wall_utc=datetime.now(timezone.utc).isoformat(),reason=reason))
        self.status.publish(String(data=json.dumps({'state':state,'reason':reason,'time':self.sim})))
        self.log('[DAY8] '+state+': '+reason);self.write()

    def approve(self,msg):
        if msg.data.strip()=='abort':self.finish(False,'OPERATOR_ABORT');return
        expected={'WAIT_APPROVAL_CLEAR_VIEW':'clear_view',
                  'WAIT_APPROVAL_PRE_PLACE':'pre_place','WAIT_APPROVAL_LOWER':'lower',
                  'WAIT_APPROVAL_RELEASE':'release','WAIT_APPROVAL_RETRACT':'retract'}
        if self.manual and self.state in expected and msg.data.strip()==expected[self.state]:
            self.pending=msg.data.strip()

    def request(self,stage):
        self.pending=None;self.approval_start_wall=time.monotonic()
        self.transition('WAIT_APPROVAL_'+stage.upper(),('approve '+stage.lower()) if self.manual else 'automatic stage gate')

    def capture_book_only(self):
        if self.rgb_stamp is None or self.rgb_stamp==self.precheck_rgb_stamp:
            return False
        self.precheck_rgb_stamp=self.rgb_stamp
        ds,depth=min(self.depths,key=lambda x:abs(x[0]-self.rgb_stamp))
        skew=abs(ds-self.rgb_stamp)
        if (skew>self.p['max_rgb_depth_skew_sec']
                or not self.fresh(ds,.55)
                or not self.fresh(self.rgb_stamp,.55)):
            raise GeometryError('HELD_BOOK_RGB_DEPTH_NOT_FRESH_OR_SYNCHRONIZED')
        registration=self.matrix(
            self.rgb_info.header.frame_id,
            self.depth_info.header.frame_id,
            ds)
        T=self.matrix(
            'base_footprint',
            self.depth_info.header.frame_id,
            ds)
        bp,_=masked_cloud(
            colour_mask(self.rgb,self.colour),
            depth,
            modelK(self.rgb_info),
            modelK(self.depth_info),
            T,
            step=1,
            depth_to_rgb=registration)
        observation=validate_book_cloud(
            bp,self.tip(),self.Tattach)
        self.book_observation=observation
        self.sensor_proof.update({
            'pre_clear_view_book_observed':True,
            'pre_clear_view_rgb_stamp_sec':self.rgb_stamp,
            'pre_clear_view_depth_stamp_sec':ds,
            'pre_clear_view_rgb_depth_skew_sec':skew,
            'tf_latest_fallbacks':self.tf_latest_fallbacks,
        })
        return True

    def capture(self):
        if self.rgb_stamp is None or self.rgb_stamp==self.processed_rgb:return False
        self.processed_rgb=self.rgb_stamp
        ds,depth=min(self.depths,key=lambda x:abs(x[0]-self.rgb_stamp))
        skew=abs(ds-self.rgb_stamp)
        if skew>self.p['max_rgb_depth_skew_sec'] or not self.fresh(ds,.55) or not self.fresh(self.rgb_stamp,.55):
            raise GeometryError('RGB_DEPTH_NOT_FRESH_OR_SYNCHRONIZED')
        # Use the actual RGB/depth extrinsics, including a nonzero sensor baseline.
        registration=self.matrix(self.rgb_info.header.frame_id,self.depth_info.header.frame_id,ds)
        T=self.matrix('base_footprint',self.depth_info.header.frame_id,ds)
        p,pix=masked_cloud(red_mask(self.rgb),depth,modelK(self.rgb_info),modelK(self.depth_info),T,depth_to_rgb=registration)
        heldT=self.tip()@self.Tattach
        local=apply(np.linalg.inv(heldT),p)
        p=p[~(np.abs(local)<np.array([.08,.01,.125])+.07).all(1)]
        near=vector_odom_to_base(self.day7['locked_bin_odom_xy'],self.odom[:2],self.odom[2])
        scene=fit_bin(p,self.profile,near)
        bp,bpix=masked_cloud(colour_mask(self.rgb,self.colour),depth,modelK(self.rgb_info),modelK(self.depth_info),T,step=1,depth_to_rgb=registration)
        observation=validate_book_cloud(bp,self.tip(),self.Tattach)
        scene.update(rgb_stamp=self.rgb_stamp,depth_stamp=ds,rgb_depth_skew_sec=skew,
                     measured_at_sim=self.sim,source='live_RGB_depth_TF_rim_plus_official_shape',
                     rgb_frame=self.rgb_info.header.frame_id,depth_frame=self.depth_info.header.frame_id,
                     source_odom=list(self.odom),mesh_sha256=sha(self.mesh_path))
        if self.scenes:
            prior=self.scenes[-1]
            if np.linalg.norm(np.array(prior['base_T_bin'])[:3,3]-np.array(scene['base_T_bin'])[:3,3])>.025:
                self.scenes=[]
        self.scenes.append(scene);self.book_observation=observation
        self.sensor_proof={'max_rgb_depth_skew_sec':skew,'depth_age_sec':self.sim-ds,
                           'live_tip_identity_observations':len(self.ledger.tip_samples),
                           'depth_to_rgb_transform':registration.tolist()}
        if len(self.scenes)>=self.p['geometry_confirm_frames']:
            self.scene=scene;self.scene['confirming_frames']=len(self.scenes)
            self.evidence_image('bin_geometry');return True
        return False

    def scene_still_consistent(self,label):
        """Recheck visible red support surfaces without changing the locked pose."""
        if not self.fresh(self.rgb_stamp,.55) or not self.depths:
            raise GeometryError('BIN_RECHECK_STALE_RGB')
        ds,depth=min(self.depths,key=lambda item:abs(item[0]-self.rgb_stamp))
        if abs(ds-self.rgb_stamp)>.20 or not self.fresh(ds,.55):
            raise GeometryError('BIN_RECHECK_STALE_DEPTH')
        T=self.matrix('base_footprint',self.depth_info.header.frame_id,ds)
        registration=self.matrix(self.rgb_info.header.frame_id,self.depth_info.header.frame_id,ds)
        p,_=masked_cloud(red_mask(self.rgb),depth,modelK(self.rgb_info),modelK(self.depth_info),T,depth_to_rgb=registration)
        relative=apply(np.linalg.inv(self.tip()@self.Tattach),p)
        p=p[~(np.abs(relative)<np.array([.08,.01,.125])+.04).all(1)]
        local=apply(np.linalg.inv(np.array(self.scene['base_T_bin'])),p)
        ox,oy=self.scene['outer_half_lengths_m'];ix,iy=self.scene['inner_half_lengths_m']
        floor=self.scene['floor_z_m']-self.scene['rim_z_m']
        use=(np.abs(local[:,0])<ox+.06)&(np.abs(local[:,1])<oy+.06)&(local[:,2]>floor-.025)&(local[:,2]<.025)
        q=local[use]
        if len(q)<80:raise GeometryError('BIN_RECHECK_TOO_FEW_VISIBLE_SURFACE_POINTS')
        residual=np.min(np.column_stack((np.abs(np.abs(q[:,0])-ox),np.abs(np.abs(q[:,0])-ix),
                    np.abs(np.abs(q[:,1])-oy),np.abs(np.abs(q[:,1])-iy),
                    np.abs(q[:,2]-floor),np.abs(q[:,2]))),axis=1)
        value=float(np.percentile(residual,80))
        if value>self.p['bin_pose_drift_tolerance_m']:
            raise GeometryError('BIN_MOVED_OR_GEOMETRY_INCONSISTENT')
        self.scene_live_checks.append(dict(stage=label,sim_time=self.sim,rgb_stamp=self.rgb_stamp,
                  depth_stamp=ds,points=int(len(q)),surface_residual80_m=value,passed=True))

    def evidence_image(self,label):
        if self.rgb is None:return
        image=self.rgb.copy()
        try:
            camera_T_base=self.matrix(self.rgb_info.header.frame_id,'base_footprint',self.rgb_stamp)
            fx,fy,cx,cy=modelK(self.rgb_info)
            def project(points):
                p=apply(camera_T_base,points)
                if np.any(p[:,2]<=.05):raise GeometryError('overlay behind camera')
                return np.rint(np.column_stack((fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy))).astype(np.int32)
            if self.scene:
                hx,hy=self.scene['outer_half_lengths_m']
                outline=apply(np.array(self.scene['base_T_bin']),[[-hx,-hy,0],[hx,-hy,0],[hx,hy,0],[-hx,hy,0]])
                cv2.polylines(image,[project(outline)],True,(0,255,0),2)
            if self.plan:
                for name,key in (('pre','pre_place_xyz_m'),('lower','lower_place_xyz_m'),('retract','retract_xyz_m')):
                    x,y=project([self.plan[key]])[0]
                    cv2.circle(image,(int(x),int(y)),4,(0,255,255),-1)
                    cv2.putText(image,'planned '+name,(int(x)+5,int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        except Exception:
            pass  # Preserve the original live frame when an overlay is outside view.
        lines=['KU SPARCy - DAY 8 '+label,'state: '+self.state,
                'arm: '+self.arm+' | target: '+self.colour,'sim: '+str(self.sim),
                'UTC: '+datetime.now(timezone.utc).isoformat(),
                'book/bin support samples: '+str(len(self.ledger.support_samples)),
                'robot/bin: '+str(self.ledger.robot_bin)]
        for i,line in enumerate(lines):cv2.putText(image,line,(8,20+18*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        folder=Path(self.p['image_output_dir']);folder.mkdir(parents=True,exist_ok=True)
        path=folder/('day8_'+label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.png')
        if not cv2.imwrite(str(path),image):raise RuntimeError('EVIDENCE_IMAGE_WRITE_FAILED')
        self.evidence[label]=str(path)
        self.evidence_stamps[label]=dict(rgb_stamp_sec=self.rgb_stamp,saved_sim_sec=self.sim)

    def head_survey(self):
        if self.op=='plan':raise RuntimeError('PLAN_ONLY_MUST_NOT_COMMAND_HEAD')
        tilts=self.p['head_survey_tilts_rad']
        offsets=self.p['head_survey_pan_offsets_rad']
        if self.head_index>=len(tilts):
            self.finish(False,'BIN_GEOMETRY_NOT_OBSERVABLE_FROM_CLEAR_VIEW_AND_SAFE_HEAD_POSES')
            return
        near=vector_odom_to_base(
            self.day7['locked_bin_odom_xy'],
            self.odom[:2],
            self.odom[2])
        base_pan=math.atan2(near[1],near[0])
        pan=float(np.clip(
            base_pan+float(offsets[self.head_index]),
            -.70,.70))
        tilt=float(tilts[self.head_index])
        self.head_target=(pan,tilt)
        self.head_index+=1
        self.head_pub.publish(
            self.trajectory(
                ['head_1_joint','head_2_joint'],
                [(self.head_target,self.p['head_motion_sec'])]))
        self.command_counts['head']+=1
        self.settle_count=0
        self.settle_stamp=None
        self.transition(
            'WAIT_SURVEY_HEAD',
            'bounded 2-D pan/tilt view for rim geometry')

    def settled(self,names,values,tolerance):
        if self.joint_stamp==self.settle_stamp:return self.settle_count>=self.p['settle_samples']
        self.settle_stamp=self.joint_stamp
        error=max(abs(self.joints[n]-q) for n,q in zip(names,values))
        self.settle_count=self.settle_count+1 if error<=tolerance else 0
        return self.settle_count>=self.p['settle_samples']

    def begin_clear_view_plan(self):
        R=rpy_matrix(
            self.day5['grasp_plan']['candidate_plans'][self.arm]['target_rpy_base'])
        near=np.asarray(
            vector_odom_to_base(
                self.day7['locked_bin_odom_xy'],
                self.odom[:2],
                self.odom[2]),
            dtype=float)
        settings={
            'book_margin_m':max(
                float(self.p['book_margin_m']),
                float(self.book_observation['uncertainty_margin_m'])),
            'robot_margin_m':float(self.p['robot_margin_m']),
            'cart_step_m':float(self.p['cart_step_m']),
            'max_joint_speed_radps':float(self.p['max_joint_speed_radps']),
            'gripper_hold_position_m':float(self.p['gripper_hold_position_m']),
            'clear_view_back_m':float(self.p['clear_view_back_m']),
            'clear_view_lateral_m':float(self.p['clear_view_lateral_m']),
            'clear_view_up_m':float(self.p['clear_view_up_m']),
            'clear_view_min_cross_track_m':float(
                self.p['clear_view_min_cross_track_m']),
            'clear_view_speed_mps':float(self.p['clear_view_speed_mps']),
            'clear_view_max_joint_delta_rad':float(
                self.p['clear_view_max_joint_delta_rad']),
        }
        self.clear_view_snapshot=dict(self.joints)
        envelopes=CollisionGeometry(
            self.urdf_path,self.model,self.arm,share)
        self.planning_started_wall=time.monotonic()
        self.future=self.pool.submit(
            build_clear_view_plan,
            self.model,
            envelopes,
            dict(self.joints),
            self.arm,
            R,
            self.Tattach,
            near,
            settings)
        self.transition(
            'PLAN_CLEAR_VIEW',
            'plan preserved-orientation arm-only motion away from camera/bin sightline')

    def start_clear_view(self):
        seg=self.clear_view_plan['segment']
        if not seg['waypoints']:
            self.clear_view_done=True
            self.scenes=[]
            self.processed_rgb=None
            self.head_index=0
            self.transition(
                'MEASURE_BIN',
                'clear-view geometry already satisfied')
            return
        current=np.array([self.joints[n] for n in self.names])
        start=np.array(seg['start_positions_rad'])
        if max(abs(current-start))>.06:
            self.finish(False,'CLEAR_VIEW_START_JOINTS_MISMATCH')
            return
        self.motion_name='clear_view'
        self.motion_q=np.array(seg['waypoints'][-1]['positions'])
        pts=[(current.tolist(),.05)]
        pts += [
            (w['positions'],float(w['time_sec'])+.05)
            for w in seg['waypoints']
        ]
        self.arm_pub.publish(
            self.trajectory(self.names,pts))
        self.command_counts['arm']+=1
        self.motion_started=self.sim
        self.motion_deadline=(
            float(seg['duration_sec'])
            + .05
            + self.p['stage_timeout_margin_sec'])
        self.settle_count=0
        self.settle_stamp=None
        self.transition(
            'MOVE_CLEAR_VIEW',
            'screened arm-only carried-book move; base remains zero')

    def begin_plan(self):
        self.snapshot_joints = dict(self.joints)

        R = rpy_matrix(
            self.day5[
                'grasp_plan'
            ][
                'candidate_plans'
            ][
                self.arm
            ][
                'target_rpy_base'
            ]
        )

        self.envelopes = CollisionGeometry(
            self.urdf_path,
            self.model,
            self.arm,
            share,
        )

        settings = {
            k: self.p[k]
            for k in (
                'book_margin_m',
                'robot_margin_m',
                'rim_clearance_m',
                'cart_step_m',
                'lower_speed_mps',
                'preplace_speed_mps',
                'retract_speed_mps',
                'max_joint_speed_radps',
                'retract_away_m',
            )
        }

        settings['book_margin_m'] = max(
            settings['book_margin_m'],
            self.book_observation[
                'uncertainty_margin_m'
            ],
        )

        # Preserve the old no-motion plan operation.
        if self.op == 'plan':
            self.planning_started_wall = (
                time.monotonic()
            )

            self.future = self.pool.submit(
                build_placement_plan,
                self.model,
                self.envelopes,
                dict(self.joints),
                self.arm,
                R,
                self.Tattach,
                self.scene,
                settings,
            )

            self.transition(
                'PLAN_PREPLACE_LOWER_RETRACT',
                'plan-only compatibility path',
            )

            return

        # EXECUTE: calculate only the very next motion.
        settings[
            'fast_preplace_budget_sec'
        ] = float(
            self.p[
                'fast_preplace_budget_sec'
            ]
        )

        settings[
            'fast_stage_ik_iterations'
        ] = int(
            self.p[
                'fast_stage_ik_iterations'
            ]
        )

        settings[
            'fast_stage_joint_samples'
        ] = int(
            self.p[
                'fast_stage_joint_samples'
            ]
        )

        settings[
            'fast_stage_seed_retries'
        ] = int(
            self.p[
                'fast_stage_seed_retries'
            ]
        )

        # Do not make IK rediscover the 90-degree restoration.
        # FAST67 exports the exact compact-carry pose that existed
        # immediately before its screened gravity-support rotation.
        settings[
            'known_restore_positions_rad'
        ] = self.day7.get(
            'compact_carry_target_positions_rad'
        )

        self.planning_started_wall = (
            time.monotonic()
        )

        self.future = self.pool.submit(
            build_fast_preplace_plan,
            self.model,
            self.envelopes,
            dict(self.joints),
            self.arm,
            R,
            self.Tattach,
            self.scene,
            settings,
        )

        self.transition(
            'PLAN_PREPLACE_FAST',
            'plan ONLY the immediate pre-place motion; '
            'lower/retract are deferred until their live states',
        )


    def begin_lower_plan(self):
        """Plan lowering only after PREPLACE physically settled."""

        self.snapshot_joints = dict(
            self.joints
        )

        R = rpy_matrix(
            self.day5[
                'grasp_plan'
            ][
                'candidate_plans'
            ][
                self.arm
            ][
                'target_rpy_base'
            ]
        )

        settings = {
            k: self.p[k]
            for k in (
                'book_margin_m',
                'robot_margin_m',
                'rim_clearance_m',
                'cart_step_m',
                'lower_speed_mps',
                'preplace_speed_mps',
                'retract_speed_mps',
                'max_joint_speed_radps',
                'retract_away_m',
            )
        }

        settings['book_margin_m'] = max(
            settings['book_margin_m'],
            self.book_observation[
                'uncertainty_margin_m'
            ],
        )

        settings[
            'fast_lower_budget_sec'
        ] = float(
            self.p[
                'fast_lower_budget_sec'
            ]
        )

        settings[
            'fast_stage_ik_iterations'
        ] = int(
            self.p[
                'fast_stage_ik_iterations'
            ]
        )

        settings[
            'fast_stage_joint_samples'
        ] = int(
            self.p[
                'fast_stage_joint_samples'
            ]
        )

        self.planning_started_wall = (
            time.monotonic()
        )

        self.future = self.pool.submit(
            build_fast_lower_plan,
            self.model,
            self.envelopes,
            dict(self.joints),
            self.arm,
            R,
            self.Tattach,
            self.scene,
            settings,
            self.plan,
        )

        self.transition(
            'PLAN_LOWER_FAST',
            'pre-place physically settled; '
            'plan lowering from current LIVE joints only',
        )


    def screen_actual_retract(self,joints,actual):
        R=rpy_matrix(self.day5['grasp_plan']['candidate_plans'][self.arm]['target_rpy_base'])
        settings={k:self.p[k] for k in ('robot_margin_m','cart_step_m','retract_speed_mps',
                                      'max_joint_speed_radps','retract_away_m')}
        joints[self.gripper_name]=self.p['gripper_release_positions_m'][-1]
        return plan_actual_retract(self.model,self.envelopes,joints,self.arm,R,self.Tattach,
                                   self.scene,np.array(self.release_geometry),settings,self.plan)

    def start_path(self,name):
        seg=self.plan['segments'][name]
        self.motion_name=name;self.motion_q=np.array(seg['waypoints'][-1]['positions'])
        if max(abs(np.array([self.joints[n] for n in self.names])-np.array(seg['start_positions_rad'])))>.06:
            self.finish(False,'TRAJECTORY_START_JOINTS_MISMATCH:'+name);return
        pts=[([self.joints[n] for n in self.names],.05)]
        pts += [(w['positions'],float(w['time_sec'])+.05) for w in seg['waypoints']]
        self.arm_pub.publish(self.trajectory(self.names,pts));self.command_counts['arm']+=1
        self.motion_started=self.sim;self.motion_deadline=seg['duration_sec']+.05+self.p['stage_timeout_margin_sec']
        self.settle_count=0;self.settle_stamp=None
        self.transition('MOVE_'+name.upper(),name+' screened positions-only waypoints')

    def start_lower_step(self):
        waypoints=self.plan['segments']['lower_place']['waypoints']
        if self.lower_index>=len(waypoints):
            self.transition('VERIFY_SUPPORT','lower endpoint reached; support is mandatory before release');return
        w=waypoints[self.lower_index];self.lower_index+=1
        self.motion_q=np.array(w['positions']);self.motion_started=self.sim
        self.motion_duration=w['dt_sec']
        self.motion_deadline=w['dt_sec']+self.p['stage_timeout_margin_sec']
        self.arm_pub.publish(self.trajectory(self.names,[(self.motion_q,w['dt_sec'])]));self.command_counts['arm']+=1
        self.settle_count=0;self.settle_stamp=None
        self.transition('LOWERING','short vertical segment '+str(self.lower_index))

    def visual_deposit_tick(self):
        if (self.colour=='red' or self.rgb_stamp==self.processed_rgb
                or not self.fresh(self.rgb_stamp,.55) or self.retract_done is None
                or self.rgb_stamp<self.retract_done):return
        self.processed_rgb=self.rgb_stamp
        try:
            ds,depth=min(self.depths,key=lambda x:abs(x[0]-self.rgb_stamp))
            if abs(ds-self.rgb_stamp)>.20 or not self.fresh(ds,.55):return
            T=self.matrix('base_footprint',self.depth_info.header.frame_id,ds)
            registration=self.matrix(self.rgb_info.header.frame_id,self.depth_info.header.frame_id,ds)
            p,_=masked_cloud(colour_mask(self.rgb,self.colour),depth,modelK(self.rgb_info),modelK(self.depth_info),T,depth_to_rgb=registration)
            p=apply(np.linalg.inv(np.array(self.scene['base_T_bin'])),p)
            hx,hy=np.array(self.scene['inner_half_lengths_m'])-.005
            floor=self.scene['floor_z_m']-self.scene['rim_z_m']
            valid=(np.abs(p[:,0])<hx)&(np.abs(p[:,1])<hy)&(p[:,2]>floor-.02)&(p[:,2]<floor+.28)
            if np.count_nonzero(valid)>=16:
                self.visual_deposit.append({'stamp':self.rgb_stamp,'points':int(np.count_nonzero(valid)),
                                            'centroid_bin':np.median(p[valid],axis=0).tolist()})
                self.visual_deposit=self.visual_deposit[-30:]
        except Exception:pass

    def tick(self):
        if self.done:return
        try:self._tick()
        except Exception as e:self.finish(False,type(e).__name__+': '+str(e))

    def _tick(self):
        if time.monotonic()-self.last_clock_wall>self.p['clock_stall_wall_timeout_sec']:
            self.finish(False,'CLOCK_STALL_WALL_WATCHDOG');return

        # Take ownership of the retained grasp immediately.  Do not wait for
        # perception/planning readiness while the book hangs at an old target.
        if self.op != 'plan':
            self.keep_safe_hold()

        if self.state=='WAIT_INPUTS':
            if self.ready():
                reason=self.safe()
                if reason:self.finish(False,reason);return
                self.initial_other={n:self.joints[n] for n in self.other_names+['torso_lift_joint']}
                self.Tattach=book_attachment(
                    self.model,self.joints,self.day5,self.day4,self.arm)
                # Transport-independent handoff: the selected arm may arrive
                # through the old route or tomorrow's RETREAT/ROTATE/
                # LATERAL_TRANSIT/FINAL_ALIGN route.  Day 8 validates the live
                # grasp instead of requiring one historic joint vector.
                self.precheck_rgb_stamp=None

                # FAST HANDOFF:
                # FAST67 deliberately leaves the book in the
                # gravity-supported orientation.  The bin is already
                # directly in front of the robot after FINAL_BIN_ALIGN.
                # capture() validates BOTH the held book and fresh
                # live bin geometry, so there is no reason to spend
                # another ~30 seconds on clear-view planning/motion.
                self.transition(
                    'MEASURE_BIN',
                    'direct live bin + held-book measurement from '
                    'gravity-supported FAST67 handoff')
                return
            if time.monotonic()-self.wall_start>self.p['ready_wall_timeout_sec']:
                self.finish(False,'INPUTS_NOT_READY');return
            return
        reason=self.safe()
        if reason:self.finish(False,reason);return
        self.keep_safe_hold()
        if time.monotonic()-self.last_status_wall>1.0:
            self.write();self.last_status_wall=time.monotonic()
        if self.state=='VERIFY_HELD_BOOK_FOR_CLEAR_VIEW':
            try:
                if self.capture_book_only():
                    self.begin_clear_view_plan()
                    return
            except Exception as e:
                self.capture_attempts.append({
                    'time':self.sim,
                    'stage':'pre_clear_view_book',
                    'reason':str(e)})
                self.capture_attempts=self.capture_attempts[-30:]
            if self.sim-self.phase_start>self.p['geometry_timeout_sec']:
                self.finish(
                    False,
                    'HELD_BOOK_NOT_VISUALLY_VERIFIED_BEFORE_CLEAR_VIEW')
            return

        if self.state=='PLAN_CLEAR_VIEW':
            if self.future.done():
                try:
                    self.clear_view_plan=self.future.result()
                except Exception as e:
                    self.finish(
                        False,
                        'CLEAR_VIEW_PLAN_FAILED:'+str(e))
                    return
                if max(
                    abs(self.joints[n]-self.clear_view_snapshot[n])
                    for n in self.names
                )>.03:
                    self.finish(False,'ARM_MOVED_DURING_CLEAR_VIEW_PLANNING')
                    return
                seg=self.clear_view_plan['segment']
                self.log(
                    '[DAY8 CLEAR VIEW PLAN][PASS] '
                    +'cross_before='
                    +str(self.clear_view_plan['book_cross_track_before_m'])
                    +' cross_after='
                    +str(self.clear_view_plan['book_cross_track_after_m'])
                    +' max_joint_delta='
                    +str(seg['max_total_joint_delta_rad']))
                # Clear-view is an internally screened protective motion.
                # Execute immediately so the fragile grasp is not left hanging
                # while waiting for an operator gate.
                self.start_clear_view()
            elif (
                time.monotonic()-self.planning_started_wall
                > self.p['planning_wall_timeout_sec']
            ):
                self.finish(False,'CLEAR_VIEW_PLANNING_WALL_TIMEOUT')
            return

        if self.state=='MOVE_CLEAR_VIEW':
            seg=self.clear_view_plan['segment']
            elapsed=self.sim-self.motion_started
            if (
                self.settled(
                    self.names,
                    self.motion_q,
                    self.p['joint_tolerance_rad'])
                and elapsed>=seg['duration_sec']
            ):
                self.clear_view_done=True
                self.completed.append('clear_view')
                self.evidence_image('clear_view')
                self.scenes=[]
                self.processed_rgb=None
                self.head_index=0
                self.transition(
                    'MEASURE_BIN',
                    'clear-view arm pose settled; reconstruct bin from fresh RGB-D')
                return
            if elapsed>self.motion_deadline:
                self.finish(False,'CLEAR_VIEW_SETTLE_TIMEOUT')
            return

        if self.state=='MEASURE_BIN':
            try:
                if self.capture():
                    if self.op=='survey':self.finish(True,'live bin geometry measured; no manipulation');return
                    self.begin_plan();return
            except Exception as e:
                self.capture_attempts.append({'time':self.sim,'reason':str(e)});self.capture_attempts=self.capture_attempts[-30:]
            if self.sim-self.phase_start>self.p['geometry_timeout_sec']:
                if self.op=='plan':self.finish(False,'PLAN_VIEW_INSUFFICIENT: run operation:=survey, inspect measurement, retry plan');return
                self.scenes=[];self.head_survey()
            return
        if self.state=='WAIT_SURVEY_HEAD':
            if self.settled(['head_1_joint','head_2_joint'],self.head_target,self.p['head_tolerance_rad']):
                self.scenes=[];self.processed_rgb=self.rgb_stamp
                self.transition('MEASURE_BIN','head settled; measure only fresh frames');return
            if self.sim-self.phase_start>self.p['head_timeout_sec']:self.finish(False,'SURVEY_HEAD_TIMEOUT')
            return
        if self.state=='PLAN_PREPLACE_LOWER_RETRACT':
            if self.future.done():
                self.plan=self.future.result()
                if max(abs(self.joints[n]-v) for n,v in self.snapshot_joints.items() if n in self.names)>.03:
                    self.finish(False,'ARM_MOVED_DURING_PLANNING');return
                self.evidence_image('plan')
                self.log('[DAY8 PLACEMENT PLAN][PASS]')
                for name,seg in self.plan['segments'].items():
                    self.log(name+': pos_error='+str(seg['ik_position_error_m'])+' rot_error='+str(seg['ik_orientation_error_rad'])+
                             ' max_joint_delta='+str(seg['max_total_joint_delta_rad'])+' screened=True')
                if self.op=='plan':
                    self.finish(
                        True,
                        'fast placement trajectories valid; commanded nothing')
                    return

                if self.manual:
                    self.request('PRE_PLACE')
                else:
                    # No upright waiting period: start immediately.
                    self.scene_still_consistent('pre_place')
                    self.start_path('pre_place')
            elif time.monotonic()-self.planning_started_wall>self.p['planning_wall_timeout_sec']:
                self.finish(False,'PLANNING_INFRASTRUCTURE_WALL_TIMEOUT')
            return
        if self.state == 'PLAN_PREPLACE_FAST':
            if self.future.done():
                try:
                    self.plan = self.future.result()
                except Exception as exc:
                    self.finish(
                        False,
                        'FAST_PREPLACE_PLANNING_FAILED: '
                        + str(exc),
                    )
                    return

                if max(
                    abs(
                        self.joints[n]
                        - self.snapshot_joints[n]
                    )
                    for n in self.names
                ) > .03:
                    self.finish(
                        False,
                        'ARM_MOVED_DURING_FAST_PREPLACE_PLANNING',
                    )
                    return

                seg = self.plan[
                    'segments'
                ][
                    'pre_place'
                ]

                self.log(
                    '[DAY8 FAST PREPLACE PLAN][PASS] '
                    'wall='
                    + str(
                        self.plan.get(
                            'preplace_planning_wall_sec'
                        )
                    )
                    + ' pos_error='
                    + str(
                        seg[
                            'ik_position_error_m'
                        ]
                    )
                    + ' rot_error='
                    + str(
                        seg[
                            'ik_orientation_error_rad'
                        ]
                    )
                )

                self.evidence_image(
                    'plan'
                )

                self.request(
                    'PRE_PLACE'
                )

            elif (
                time.monotonic()
                - self.planning_started_wall
                > self.p[
                    'planning_wall_timeout_sec'
                ]
            ):
                self.finish(
                    False,
                    'FAST_PREPLACE_WALL_TIMEOUT',
                )

            return

        if self.state == 'PLAN_LOWER_FAST':
            if self.future.done():
                try:
                    result = self.future.result()
                except Exception as exc:
                    self.finish(
                        False,
                        'FAST_LOWER_PLANNING_FAILED: '
                        + str(exc),
                    )
                    return

                if max(
                    abs(
                        self.joints[n]
                        - self.snapshot_joints[n]
                    )
                    for n in self.names
                ) > .03:
                    self.finish(
                        False,
                        'ARM_MOVED_DURING_FAST_LOWER_PLANNING',
                    )
                    return

                self.plan[
                    'segments'
                ][
                    'lower_place'
                ] = result[
                    'segment'
                ]

                self.log(
                    '[DAY8 FAST LOWER PLAN][PASS] '
                    'wall='
                    + str(
                        result[
                            'planning_wall_sec'
                        ]
                    )
                )

                # Existing release/support logic begins here.
                self.request(
                    'LOWER'
                )

            elif (
                time.monotonic()
                - self.planning_started_wall
                > self.p[
                    'planning_wall_timeout_sec'
                ]
            ):
                self.finish(
                    False,
                    'FAST_LOWER_WALL_TIMEOUT',
                )

            return

        if self.state.startswith('WAIT_APPROVAL_'):
            expected={'WAIT_APPROVAL_CLEAR_VIEW':'clear_view',
                      'WAIT_APPROVAL_PRE_PLACE':'pre_place','WAIT_APPROVAL_LOWER':'lower',
                      'WAIT_APPROVAL_RELEASE':'release','WAIT_APPROVAL_RETRACT':'retract'}[self.state]
            if self.manual and self.pending!=expected:
                if time.monotonic()-self.approval_start_wall>self.p['approval_wall_timeout_sec']:
                    self.finish(False,'APPROVAL_TIMEOUT')
                return
            self.pending=None
            if expected=='clear_view':
                self.start_clear_view()
            elif expected=='pre_place':
                self.scene_still_consistent('pre_place');self.start_path('pre_place')
            elif expected=='lower':
                self.scene_still_consistent('lower');self.lower_index=0;self.start_lower_step()
            elif expected=='release':
                if not self.support_verified or not self.ledger.support_window(self.sim,.4,after=self.first_book_bin_time):
                    self.finish(False,'SUPPORT_LOST_BEFORE_RELEASE');return
                if not supported_geometry(self.tip(),self.Tattach,self.scene):
                    self.finish(False,'BOOK_OUTSIDE_INTERIOR_BEFORE_RELEASE');return
                self.scene_still_consistent('release')
                self.release_started=self.sim;self.release_geometry=(self.tip()@self.Tattach).tolist()
                self.release_index=0;self.release_step_start=None
                self.transition('OPEN_STEPWISE','intentional release starts; fingertip loss now expected')
            elif expected=='retract':
                # Contact may stop the lowering a few millimetres before the
                # nominal floor endpoint. Re-screen the ENTIRE empty-hand path
                # from the actual joints against the actual deposited-book pose.
                # Solve new IK from actual open-hand joints; no stale join.
                actual=np.array([self.joints[n] for n in self.names])
                self.snapshot_joints=dict(self.joints)
                self.future=self.pool.submit(self.screen_actual_retract,dict(self.joints),actual)
                self.planning_started_wall=time.monotonic()
                self.transition('REPLAN_RETRACT','re-screen complete empty-gripper path against deposited book')
            return
        if self.state=='REPLAN_RETRACT':
            if self.future.done():
                self.retract_rescreen=self.future.result()
                if max(abs(self.joints[n]-self.snapshot_joints[n]) for n in self.names)>.03:
                    self.finish(False,'ARM_MOVED_DURING_RETRACT_RESCREEN');return
                self.plan['segments']['retract']=self.retract_rescreen['segment']
                self.start_path('retract')
            elif time.monotonic()-self.planning_started_wall>self.p['planning_wall_timeout_sec']:
                self.finish(False,'RETRACT_RESCREEN_WALL_TIMEOUT')
            return
        if self.state in ('MOVE_PRE_PLACE','MOVE_RETRACT'):
            elapsed=self.sim-self.motion_started
            if self.settled(self.names,self.motion_q,self.p['joint_tolerance_rad']) and elapsed>=self.plan['segments'][self.motion_name]['duration_sec']:
                stage=self.motion_name;self.completed.append(stage);self.evidence_image(stage)
                if stage=='pre_place':
                    lo,hi=book_bounds_in_bin(
                        self.tip(),
                        self.Tattach,
                        self.scene,
                    )

                    if lo[2] < .025:
                        self.finish(
                            False,
                            'BOOK_NOT_ABOVE_RIM_AT_PREPLACE',
                        )
                        return

                    # Do not use a trajectory predicted before
                    # pre-place. Replan the lowering NOW from
                    # the actual settled joint state.
                    self.begin_lower_plan()
                else:
                    self.retract_done=self.sim;self.processed_rgb=None
                    self.transition('VERIFY_DEPOSIT','book must remain supported after empty gripper withdrew')
                return
            if elapsed>self.motion_deadline:self.finish(False,self.motion_name.upper()+'_SETTLE_TIMEOUT')
            return
        if self.state=='LOWERING':
            if self.lower_contact_seen:return
            if self.settled(self.names,self.motion_q,self.p['joint_tolerance_rad']) and self.sim-self.motion_started>=self.motion_duration:
                self.start_lower_step();return
            if self.sim-self.motion_started>self.motion_deadline:self.finish(False,'LOWER_SEGMENT_TIMEOUT')
            return
        if self.state=='SETTLE_SUPPORT':
            if self.settled(self.names,self.motion_q,self.p['joint_tolerance_rad']):
                self.transition('VERIFY_SUPPORT','arm settled at contact; require persistent support')
            elif self.sim-self.phase_start>4:self.finish(False,'CONTACT_STOP_DID_NOT_SETTLE')
            return
        if self.state=='VERIFY_SUPPORT':
            if self.ledger.support_window(self.sim,self.p['support_confirm_sec'],self.p['support_fresh_sec'],after=self.first_book_bin_time or self.sim):
                if not supported_geometry(self.tip(),self.Tattach,self.scene):self.finish(False,'CONTACT_IS_NOT_INTERIOR_FLOOR_SUPPORT');return
                self.support_verified=True;self.completed.append('lower_place');self.evidence_image('supported')
                self.request('RELEASE');return
            if self.sim-self.phase_start>3:self.finish(False,'NO_PERSISTENT_BOOK_BIN_SUPPORT: gripper stays closed')
            return
        if self.state=='OPEN_STEPWISE':
            values=self.p['gripper_release_positions_m']
            if self.release_index>=len(values):
                self.release_complete=self.sim;self.completed.append('open')
                self.transition('VERIFY_RELEASE','open reached; require support and quiet fingertips');return
            target=values[self.release_index]
            if self.release_step_start is None:
                self.commanded_gripper=float(target)
                self.grip_pub.publish(self.trajectory([self.gripper_name],[([target],self.p['gripper_release_step_sec'])]))
                self.command_counts['gripper']+=1;self.last_grip_publish=self.sim
                self.release_step_start=self.sim;self.settle_count=0;self.settle_stamp=None
            # Do not overwrite a slow opening trajectory with a 0.2 s hold.
            elapsed=self.sim-self.release_step_start
            if elapsed>=self.p['gripper_release_step_sec'] and abs(self.joints[self.gripper_name]-target)<=self.p['gripper_open_tolerance_m']:
                self.release_trace.append({'time':self.sim,'commanded':target,'actual':self.joints[self.gripper_name]})
                self.release_index+=1;self.release_step_start=None
            elif elapsed>self.p['gripper_release_step_sec']+3:self.finish(False,'GRIPPER_OPEN_STEP_FAILED')
            return
        if self.state=='VERIFY_RELEASE':
            quiet=self.ledger.tip_last is not None and self.sim-self.ledger.tip_last>=self.p['fingertip_quiet_sec']
            support=self.ledger.support_window(self.sim,self.p['release_verify_sec'],self.p['support_fresh_sec'],after=self.release_complete)
            if quiet and support and abs(self.joints[self.gripper_name]-self.p['gripper_release_positions_m'][-1])<=self.p['gripper_open_tolerance_m']:
                self.release_verified=True;self.evidence_image('released');self.request('RETRACT');return
            if self.sim-self.phase_start>self.p['deposit_timeout_sec']:self.finish(False,'RELEASE_NOT_CONFIRMED: no retract')
            return
        if self.state=='VERIFY_DEPOSIT':
            self.visual_deposit_tick()
            support=self.ledger.support_window(self.sim,self.p['deposit_verify_sec'],self.p['support_fresh_sec'],5,after=self.retract_done)
            quiet=self.ledger.tip_last is not None and self.sim-self.ledger.tip_last>=self.p['fingertip_quiet_sec']
            live_after_retract=self.fresh(self.rgb_stamp,.55) and self.rgb_stamp>=self.retract_done
            visual=live_after_retract and (self.colour=='red' or len(self.visual_deposit)>=3)
            if support and quiet and visual:
                self.deposit_verified=True;self.completed.append('deposit');self.evidence_image('deposited')
                self.finish(True,'book supported inside bin after gradual opening and empty-arm withdrawal');return
            if self.sim-self.phase_start>self.p['deposit_timeout_sec']:self.finish(False,'DEPOSIT_PERSISTENCE_OR_VISUAL_EVIDENCE_FAILED')
            return
        self.finish(False,'UNKNOWN_STATE:'+self.state)

    def write(self):
        data=dict(day=8,operation=self.op,passed=bool(self.passed),state=self.state,reason=self.reason,
          day7_result_path=self.p['day7_result_path'],day7_result_sha256=self.source_digest,
          day7_input_passed=self.day7.get('passed'),selected_arm=self.arm,day5_selected_arm=self.day5['selected_arm'],
          target_colour=self.colour,simulation_time_sec=self.sim,wall_duration_sec=time.monotonic()-self.wall_start,
          source_day7_sim_time=self.day7['simulation_time_sec'],
          manual_approval_required=self.manual,command_counts=self.command_counts,
          gripper_public_topic=f'/gripper_{self.arm}_controller/joint_trajectory',
          gripper_command_mode='position',effort_commanded=False,raw_topic_commanded=False,
          torso_commanded=False,hidden_simulator_oracle_used=False,world_pose_queried=False,
          bin_geometry=self.scene,held_book_observation=self.book_observation,sensor_proof=self.sensor_proof,bin_live_rechecks=self.scene_live_checks,
          clear_view_plan=self.clear_view_plan,clear_view_completed=self.clear_view_done,
          placement_handoff_contract='live_endpoint_pose+live_grasp+live_bin_anchor; transport_state_machine_independent',
          plan=self.plan,actual_retract_rescreen=self.retract_rescreen,
          completed_stages=self.completed,contact_evidence=self.ledger.as_dict(),
          last_robot_contact_stream_sim_sec=self.last_contacts_rx,last_bin_contact_stream_sim_sec=self.last_bin_rx,
          max_base_drift_m=self.max_drift,max_base_yaw_drift_deg=math.degrees(self.max_yaw_drift),
          support_verified_before_open=self.support_verified,
          maximum_book_bin_penetration_m=self.max_book_bin_penetration,book_bin_depth_samples=self.book_bin_depth_samples,first_book_bin_contact_sim_sec=self.first_book_bin_time,
          release_started_sim_sec=self.release_started,release_complete_sim_sec=self.release_complete,
          release_verified=self.release_verified,gripper_open_actual=self.joints.get(self.gripper_name),
          release_trace=self.release_trace,retract_completed_sim_sec=self.retract_done,
          deposit_verified=self.deposit_verified,visual_deposit_observations=self.visual_deposit,
          red_target_visual_limitation=(self.colour=='red'),
          release_book_base_transform=self.release_geometry,evidence_images=self.evidence,evidence_timestamps=self.evidence_stamps,
          capture_diagnostics=self.capture_attempts,state_history=self.history,
          official_urdf_sha256=self.urdf_digest,configuration=self.p)
        atomic_write_json(self.result_path,data)

    def finish(self,ok,reason):
        if self.done:return
        if not ok:
            if all(n in self.joints for n in self.names):
                self.abort_positions=[self.joints[n] for n in self.names]
            self.halt_arm()
        self.keep_safe_hold();self.passed=bool(ok);self.reason=reason
        self.state='DONE' if ok else 'FAILED';self.done=True
        self.write()
        marker=('DAY8 RESULT' if self.op=='execute' else 'DAY8 '+self.op.upper())
        self.log('['+marker+']['+('PASS' if ok else 'FAIL')+'] '+reason)


def main(args=None):
    rclpy.init(args=args);node=None;code=1
    try:
        node=Day8BinPlace()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=.05)
            # Survey/execute must maintain the retained grasp even while
            # waiting for perception/state-machine readiness.  Plan remains
            # command-free.
            if node.op != 'plan':
                node.keep_safe_hold()
        code=0 if node.passed else 1
    except KeyboardInterrupt:
        if node:node.finish(False,'OPERATOR_INTERRUPT')
    except Exception as e:
        if node:node.finish(False,str(e))
        else:print('[DAY8 RESULT][FAIL] '+str(e))
    finally:
        if node:
            if not node.passed:
                until=time.monotonic()+.40
                while rclpy.ok() and time.monotonic()<until:
                    node.keep_safe_hold()
                    if node.arm_pub is not None and node.abort_positions is not None:
                        node.arm_pub.publish(node.trajectory(node.names,[(node.abort_positions,.25)]))
                        node.command_counts['arm']+=1
                    rclpy.spin_once(node,timeout_sec=.03)
                node.write()
            node.pool.shutdown(wait=False,cancel_futures=True)
            node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    raise SystemExit(code)

if __name__=='__main__':main()
