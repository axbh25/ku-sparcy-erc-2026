#!/usr/bin/env python3
"""Static result/installed URDF contract. Does not assert a live world match."""
import argparse,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from ku_sparcy_erc.day8_geometry import official_bin_profile

p=argparse.ArgumentParser();p.add_argument('day7_result');a=p.parse_args()
d=json.loads(Path(a.day7_result).read_text());errors=[]
def check(label,ok):
    print('['+label+']['+('PASS' if ok else 'FAIL')+']')
    if not ok:errors.append(label)
check('DAY7 SOURCE PASSED',d.get('day')==7 and d.get('passed') is True)
required=['day5_result_path','final_odom_xy','final_yaw_rad','bin_detection','bin_geometry',
 'locked_bin_odom_xy','simulation_time_sec','carry_raise_target_positions_rad',
 'carry_raise_completed','rotation_alignment_completed','bin_detection_uses_live_rgb_depth_tf']
check('DAY7 SOURCE SCHEMA',all(k in d for k in required))
d5=json.loads(Path(d['day5_result_path']).read_text())
d4=json.loads(Path(d5['source_result_path']).read_text())
check('ARM CONSISTENCY',d['selected_arm']==d5['selected_arm'] and d5.get('retention_verified') is True)
check('FRESH-SOURCE CHAIN EXISTS',bool(d4.get('target_book_geometry')) and d4.get('passed') is True)
share=Path(get_package_share_directory('erc_description'))
r=ET.parse(share/'urdf/tiago_pro.urdf').getroot()
for side in ['left','right']:
    j=next(e for e in r.findall('.//ros2_control/joint') if e.get('name')==f'gripper_{side}_finger_joint')
    check(side.upper()+' GRIPPER POSITION ONLY',[e.get('name') for e in j.findall('command_interface')]==['position'])
j=next(e for e in r.findall('joint') if e.get('name')=='head_2_joint')
print('head_2_joint hard limits:',j.find('limit').attrib)
profile=official_bin_profile(share/'models/collection_bin/meshes/erc_base_collection_bin.STL')
print('Installed bin shape, no arena pose:',json.dumps(profile,indent=2))
check('OFFICIAL BIN SHAPE',profile['source']=='official_local_collision_mesh_shape_only')
print('Day 7 selected arm:',d['selected_arm'])
print('Day 7 finish simulation time:',d['simulation_time_sec'])
print('Day 7 odometry:',d['final_odom_xy'],d['final_yaw_rad'])
print('Day 7 sensed bin:',d['locked_bin_odom_xy'])
print('Day 7 high/compact joints:',d['carry_raise_target_positions_rad'])
print('[LIVE WORLD MATCH][NOT YET CHECKED] plan node checks current sensors; do not reuse results across reset.')
print('[DAY8 INPUT CONTRACT]['+('FAIL' if errors else 'PASS')+']')
sys.exit(bool(errors))
