#!/usr/bin/env python3
"""Pure geometry/contact/IK tests; does NOT certify a real TIAGo trajectory."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ku_sparcy_erc.day8_geometry import (GeometryError,transform,apply,box_corners,
    official_bin_profile,fit_bin,book_bounds_in_bin,supported_geometry,validate_book_cloud,masked_cloud)
from ku_sparcy_erc.day8_contacts import ContactLedger
from ku_sparcy_erc.day8_planner import sat_gap,CollisionGeometry,build_placement_plan,plan_actual_retract
from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel


TIP='tiago_pro::gripper_right_fingertip_left_link::collision'
OTHER='opaque_object::solid::collision'
BIN='erc_collection_bin::collection_bin_base_link::collision'
TABLE='erc_table::table_base_link::collision'


def cube_triangles(lo,hi):
    v=box_corners(lo,hi)
    # Each face includes the two triangles of an axis-aligned box.
    faces=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    return np.array([v[list(t)] for a,b,c,d in faces for t in ((a,b,c),(a,c,d))])


def write_stl(path,triangles):
    with open(path,'w') as f:
        f.write('solid test\n')
        for tri in triangles:
            f.write('facet normal 0 0 0\nouter loop\n')
            for v in tri:f.write('vertex '+' '.join(map(str,v))+'\n')
            f.write('endloop\nendfacet\n')
        f.write('endsolid test\n')


def bin_fixture(folder):
    boxes=[([-.25,-.155,0],[.25,.155,.010]),
           ([-.25,-.155,.010],[-.238,.155,.21]),
           ([.238,-.155,.010],[.25,.155,.21]),
           ([-.238,-.155,.010],[.238,-.143,.21]),
           ([-.238,.143,.010],[.238,.155,.21])]
    tris=np.concatenate([cube_triangles(*b) for b in boxes])
    path=Path(folder)/'bin.stl';write_stl(path,tris)
    return path


def live_bin_points():
    p=[]
    for z in np.linspace(.8,1.,40):
        for x in np.linspace(-.25,.25,81):
            for y in (-.155,.155):p.append([1.+x,y,z])
        for y in np.linspace(-.155,.155,61):
            for x in (-.25,.25):p.append([1.+x,y,z])
    for x in np.linspace(-.20,.20,50):
        for y in np.linspace(-.12,.12,30):p.append([1.+x,y,.8])
    return np.array(p)


def scene_fixture():
    return dict(base_T_bin=transform(t=[1.,0.,1.]).tolist(),rim_z_m=1.,floor_z_m=.8,
                table_z_m=.79,outer_half_lengths_m=[.25,.155],inner_half_lengths_m=[.238,.143])


class ContactsTest(unittest.TestCase):
    def test_table_support_never_counts_as_book(self):
        c=ContactLedger('right')
        for t in np.arange(0,3,.1):self.assertIsNone(c.feed((BIN,TABLE),float(t),'WAIT_INPUTS'))
        self.assertEqual(c.ignored_support,30);self.assertEqual(c.support_samples,[])

    def test_phase_policy_and_identity(self):
        c=ContactLedger('right');c.feed((TIP,OTHER),1.,'WAIT_INPUTS')
        self.assertEqual(c.held_id,OTHER)
        self.assertEqual(c.feed((BIN,OTHER),1.1,'MOVE_PRE_PLACE',False),'PREMATURE_HELD_OBJECT_BIN_CONTACT')
        d=ContactLedger('right');d.feed((TIP,OTHER),1.,'WAIT_INPUTS')
        self.assertIsNone(d.feed((BIN,OTHER),2.,'LOWERING',True))
        self.assertEqual(d.feed((BIN,'a_different_object::collision'),2.2,'LOWERING',True),'UNIDENTIFIED_OBJECT_BIN_CONTACT')
        for phase in ('LOWERING','OPEN_STEPWISE','VERIFY_DEPOSIT'):
            e=ContactLedger('right');e.feed((TIP,OTHER),1.,'WAIT_INPUTS')
            self.assertEqual(e.feed((BIN,TIP),3.,phase,True),'ROBOT_BIN_CONTACT')

    def test_persistence_cannot_use_single_or_duplicate_frame(self):
        c=ContactLedger('right');c.feed((TIP,OTHER),0.,'WAIT_INPUTS')
        for _ in range(100):c.feed((BIN,OTHER),1.,'LOWERING',True)
        self.assertFalse(c.support_window(1.,.4))
        for t in np.arange(1.1,2.01,.1):c.feed((BIN,OTHER),round(float(t),3),'VERIFY_SUPPORT',True)
        self.assertTrue(c.support_window(2.,.4))
        self.assertFalse(c.support_window(3.,.4))
        self.assertFalse(c.support_window(2.,.4,after=1.9))


class GeometryTest(unittest.TestCase):
    def test_local_mesh_and_live_fit(self):
        with tempfile.TemporaryDirectory() as d:
            profile=official_bin_profile(bin_fixture(d))
        self.assertAlmostEqual(profile['interior_depth_m'],.20,places=4)
        self.assertEqual(profile['source'],'official_local_collision_mesh_shape_only')
        scene=fit_bin(live_bin_points(),profile,[1.,0.])
        self.assertTrue(scene['live_geometry']);self.assertTrue(scene['floor_directly_observed'])
        self.assertAlmostEqual(scene['rim_z_m'],1.,places=3)
        self.assertAlmostEqual(scene['floor_z_m'],.8,places=3)
        with self.assertRaises(GeometryError):
            fit_bin(live_bin_points()[live_bin_points()[:,1]>.12],profile,[1.,0.])

    def test_noncoincident_rgb_depth_registration(self):
        depth=np.ones((20,30),float);mask=np.zeros((20,30),bool);mask[:,11:16]=True
        K=[100.,100.,15.,10.];reg=transform(t=[.02,0,0])
        p,uv=masked_cloud(mask,depth,K,K,np.eye(4),step=1,depth_to_rgb=reg)
        self.assertGreater(len(p),20)
        self.assertTrue(np.all((uv[:,0]>=11)&(uv[:,0]<16)))
        self.assertTrue(np.allclose(100*(p[:,0]+.02)/p[:,2]+15,uv[:,0]))

    def test_inside_support_not_rim(self):
        s=scene_fixture();A=transform(t=[.08,0,0])
        inside=transform(t=[.92,0,.925])
        self.assertTrue(supported_geometry(inside,A,s))
        on_rim=transform(t=[.72,0,1.125])
        self.assertFalse(supported_geometry(on_rim,A,s))

    def test_carried_book_reaches_wall_when_empty_tip_is_clear(self):
        boxT=transform(t=[.75,0,.95]);wall=transform(t=[.744,0,.895])
        gap=sat_gap(boxT,np.array([.08,.01,.125]),wall,np.array([.006,.155,.105]))
        self.assertLess(gap,0)
        empty=transform(t=[.65,0,1.15])
        self.assertGreater(sat_gap(empty,np.full(3,.01),wall,np.array([.006,.155,.105])),0)

    def test_cloud_wrong_colour_location_not_enough(self):
        T=transform(t=[.6,0,1.1]);A=transform(t=[.08,0,0])
        points=np.array([[.60,y,z] for y in np.linspace(-.008,.008,10) for z in np.linspace(1.,1.2,15)])
        self.assertGreater(validate_book_cloud(points,T,A)['consistent_colour_points'],20)
        with self.assertRaises(GeometryError):validate_book_cloud(points+[2,0,0],T,A)


class PlannerTest(unittest.TestCase):
    def test_synthetic_cartesian_chain_all_three_paths(self):
        with tempfile.TemporaryDirectory() as d:
            mesh=Path(d)/'tiny.stl';write_stl(mesh,cube_triangles([-.001]*3,[.001]*3))
            text=['<robot name="synthetic"><link name="base_footprint"/>']
            parent='base_footprint';joints={}
            for i in range(1,8):
                name=f'arm_right_{i}_joint';child=f'arm_right_{i}_link'
                kind='prismatic' if i<=3 else 'revolute'
                axis=['1 0 0','0 1 0','0 0 1'][min(i-1,2)]
                text.append(f'<link name="{child}"><collision><origin xyz="-0.03 0 0"/><geometry><mesh filename="{mesh}"/></geometry></collision></link>')
                text.append(f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                            f'<origin xyz="0 0 {1.3 if i==1 else 0.0}"/><axis xyz="{axis}"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>')
                joints[name]=[.60,0,-.05,0,0,0,0][i-1];parent=child
            text.append(f'<link name="gripper_right_grasping_link"/><joint name="tip" type="fixed"><parent link="{parent}"/>'
                        '<child link="gripper_right_grasping_link"/></joint></robot>')
            path=Path(d)/'robot.urdf';path.write_text(''.join(text));model=URDFKinematicModel.from_file(str(path))
            env=CollisionGeometry(path,model,'right',lambda p:d)
            settings=dict(book_margin_m=.012,robot_margin_m=.008,rim_clearance_m=.055,cart_step_m=.025,
                          lower_speed_mps=.018,preplace_speed_mps=.05,retract_speed_mps=.045,
                          max_joint_speed_radps=.16,retract_away_m=.10)
            plan=build_placement_plan(model,env,joints,'right',np.eye(3),transform(t=[.08,0,0]),scene_fixture(),settings)
            self.assertTrue(plan['passed']);self.assertTrue(plan['grasp_orientation_preserved'])
            for key in ('pre_place','lower_place','retract'):
                self.assertTrue(plan['segments'][key]['passed'])
                self.assertLessEqual(plan['segments'][key]['ik_position_error_m'],.0025)
                self.assertTrue(plan['segments'][key]['collision_screen']['passed'])
            self.assertAlmostEqual(plan['pre_place_xyz_m'][0],plan['lower_place_xyz_m'][0],places=6)
            self.assertAlmostEqual(plan['pre_place_xyz_m'][1],plan['lower_place_xyz_m'][1],places=6)
            low=plan['segments']['lower_place']['waypoints'][-1]['positions']
            actual=dict(joints);actual.update(zip(plan['joint_names'],low))
            actual['arm_right_3_joint']+=.001
            actual['gripper_right_finger_joint']=.04
            deposited=model.forward(model.chain('base_footprint','gripper_right_grasping_link'),actual).tip_transform@transform(t=[.08,0,0])
            fresh=plan_actual_retract(model,env,actual,'right',np.eye(3),transform(t=[.08,0,0]),scene_fixture(),deposited,settings,plan)
            self.assertTrue(fresh['new_ik_from_actual_joints'])
            self.assertTrue(fresh['actual_deposited_book_checked'])
            self.assertTrue(fresh['passed'])



if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if result.wasSuccessful():
        print('[DAY8 CONTACT POLICY][PASS]')
        print('[DAY8 LOCAL MESH AND LIVE GEOMETRY][PASS]')
        print('[DAY8 CARRIED BOOK CLEARANCE][PASS]')
        print('[DAY8 SYNTHETIC IK PATHS][PASS]')
        print('[DAY8 OFFLINE TEST][PASS]')
        print('[REAL TIAGO PLACEMENT][NOT TESTED HERE] Run the live plan-only gate.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
