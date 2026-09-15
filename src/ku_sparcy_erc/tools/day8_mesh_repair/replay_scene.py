#!/usr/bin/env python3
"""Read-only replay of captured robot/scene archives; never imports ROS."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

def load_inputs(repro,scene_root):
    repro=Path(repro); scene_root=Path(scene_root)
    sys.path.insert(0,str(repro/'team'))
    from ku_sparcy_erc.day8_geometry import transform,quat_matrix,masked_cloud,red_mask,book_attachment
    from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel,rpy_matrix
    import day8_mesh_core as core
    data={n:json.loads((repro/'run'/f'{n}.json').read_text()) for n in ('day4','day5','day7_fast','day8')}
    def read(n):return json.loads((scene_root/'current_scene'/f'{n}.json').read_text())
    rgbinfo,depinfo=read('rgb'),read('depth')
    rgb=np.fromfile(scene_root/'current_scene/rgb.bin',np.uint8).reshape(rgbinfo['height'],rgbinfo['width'],3)[:,:,::-1].copy()
    dep=np.fromfile(scene_root/'current_scene/depth.bin','<f4').reshape(depinfo['height'],depinfo['width'])
    tf=read('transforms')
    def M(k):return transform(quat_matrix(tf[k]['quaternion_xyzw']),tf[k]['translation_xyz'])
    def K(n):k=read(n)['k'];return[k[0],k[4],k[2],k[5]]
    red,pix=masked_cloud(red_mask(rgb),dep,K('rgb_info'),K('depth_info'),M('base_T_depth'),step=1,depth_to_rgb=M('rgb_T_depth'))
    white=(rgb.max(2)-rgb.min(2)<12)&(rgb.min(2)>160)
    whites,_=masked_cloud(white,dep,K('rgb_info'),K('depth_info'),M('base_T_depth'),step=1,depth_to_rgb=M('rgb_T_depth'))
    assets=core.LocalAssets(scene_root/'installed_models')
    t=time.monotonic()
    # Bin association hint from historical measured opening only, never a world pose.
    measured=core.measured_scene(red,whites,assets,np.array(data['day8']['bin_geometry']['base_T_bin'])[:2,3],budget=10.)
    print('SCENE FIT',time.monotonic()-t,measured['bin_mesh_registration'],flush=True)
    print('BIN',np.asarray(measured['base_T_bin'])[:3,3], 'TABLE',np.asarray(measured['base_T_table_top'])[:3,3],flush=True)
    model=core.FastModel(URDFKinematicModel.from_file(str(repro/'packages/erc_description/urdf/tiago_pro.urdf')))
    joints=read('joints')['positions'];arm=data['day7_fast']['selected_arm'];names=[f'arm_{arm}_{i}_joint' for i in range(1,8)]
    qs=data['day7_fast']['gravity_supported_target_positions_rad'];qr=data['day7_fast']['compact_carry_target_positions_rad']
    joints.update(zip(names,qs))
    attach=book_attachment(model,joints,data['day5'],data['day4'],arm)
    rotation=rpy_matrix(data['day5']['grasp_plan']['candidate_plans'][arm]['target_rpy_base'])
    settings=dict(data['day8']['configuration']);settings['book_margin_m']=max(settings['book_margin_m'],data['day8']['held_book_observation']['uncertainty_margin_m'])
    screen=core.MeshScreen(str(repro/'packages/erc_description/urdf/tiago_pro.urdf'),model,arm,lambda n:repro/'packages'/n,assets,joints,attach,settings)
    return dict(core=core,repro=repro,scene_root=scene_root,scene=measured,model=model,joints=joints,restore=qr,rotation=rotation,settings=settings,screen=screen,data=data,rgb=rgb,red=red,white=whites,assets=assets)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repro',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=float,default=30.);a=p.parse_args()
    d=load_inputs(a.repro,a.scene);core=d['core'];start=time.monotonic()
    plan=core.build_complete_plan(d['screen'],d['joints'],d['restore'],d['rotation'],d['scene'],d['settings'],start+a.budget)
    from ku_sparcy_erc.day8_geometry import colour_mask
    plan['replay_provenance']={'no_robot_commands':True,'not_a_Gazebo_physics_test':True,'current_scene_blue_pixels':int(colour_mask(d['rgb'],'blue').sum()),'planned_book_attachment_from_historical_grasp':True,'later_scene_is_not_historical_contact_evidence':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print('FULL OFFLINE PLAN',plan['docking_forward_m'],plan['bin_local_target_x_m'],'wall',time.monotonic()-start)
    for name,seg in plan['segments'].items(): print(name,seg['duration_sec'],'samples',seg['collision_screen']['samples'],'PE',seg['ik_position_error_m'],'RE',seg['ik_orientation_error_rad'])
    print('FAILED CANDIDATES',plan['rejected_candidates']);print('OUTPUT',a.output)
if __name__=='__main__':main()
