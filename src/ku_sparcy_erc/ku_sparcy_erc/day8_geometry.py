"""Measured bin placement geometry; no arena/world/spawn pose is read."""
from pathlib import Path
import math
import struct
import xml.etree.ElementTree as ET
import numpy as np
import cv2


class GeometryError(ValueError):
    pass


def transform(R=None, t=None):
    T=np.eye(4)
    if R is not None: T[:3,:3]=R
    if t is not None: T[:3,3]=t
    return T


def apply(T, points):
    p=np.asarray(points, dtype=float).reshape(-1,3)
    return p @ np.asarray(T)[:3,:3].T + np.asarray(T)[:3,3]


def quat_matrix(q):
    x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def box_corners(lo,hi):
    return np.array([[x,y,z] for x in (lo[0],hi[0])
                     for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])


def stl_triangles(path):
    raw=Path(path).read_bytes()
    if len(raw)>=84:
        n=struct.unpack_from('<I',raw,80)[0]
        if 84+50*n==len(raw):
            dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
            return np.frombuffer(raw,dt,count=n,offset=84)['v'].astype(float)
    try:
        vertices=[list(map(float,line.strip().split()[1:]))
                  for line in raw.decode('ascii').splitlines()
                  if line.strip().startswith('vertex ')]
        out=np.array(vertices).reshape(-1,3,3)
        if len(out): return out
    except (UnicodeError,ValueError): pass
    raise GeometryError('UNSUPPORTED_COLLISION_MESH: '+str(path))


def official_bin_profile(path):
    """Extract local shape only. Select opening side from broad floor triangles.

    The smallest AABB extent is the bin's vertical dimension. Horizontal broad
    faces identify outside bottom and inside floor; the far side is the open rim.
    Axis signs/translation in the arena are NOT read from any world file.
    """
    tris=stl_triangles(path)
    pts=tris.reshape(-1,3); lo=pts.min(0); hi=pts.max(0); size=hi-lo
    order=np.argsort(size); v=int(order[0]); w=int(order[1]); l=int(order[2])
    if not (0.12<size[v]<0.30 and 0.24<size[w]<0.40 and 0.40<size[l]<0.60):
        raise GeometryError('BIN_MESH_DIMENSIONS_NOT_EXPECTED: '+str(size))
    normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]); areas=np.linalg.norm(normals,axis=1)/2
    unit=normals/np.maximum(2*areas[:,None],1e-12)
    levels={}
    for z,a in zip(np.round(tris[:,:,v].mean(1)[np.abs(unit[:,v])>0.97],4),areas[np.abs(unit[:,v])>0.97]):
        levels[float(z)]=levels.get(float(z),0)+float(a)
    if len(levels)<3: raise GeometryError('BIN_FLOOR_LAYERS_UNRESOLVED')
    max_area=max(levels.values()); broad=sorted(z for z,a in levels.items() if a>max_area*.40)
    mid=(lo[v]+hi[v])/2; bottom_low=np.mean(broad)<mid
    floor=max(broad) if bottom_low else min(broad)
    rim=hi[v] if bottom_low else lo[v]
    floor_depth=abs(rim-floor)
    if not 0.08<floor_depth<size[v]: raise GeometryError('BIN_INTERIOR_DEPTH_UNRESOLVED')
    inner=[]
    for axis in (l,w):
        side=tris[np.abs(unit[:,axis])>0.65].reshape(-1,3)
        center=(lo[axis]+hi[axis])/2
        a=np.abs(side[:,axis]-center)
        # Ignore floor triangulation and small handle details. The smallest
        # substantial side-wall radius bounds the interior at all heights.
        a=a[a>size[axis]*0.30]
        if not len(a): raise GeometryError('BIN_INNER_WALL_UNRESOLVED')
        inner.append(float(min(np.min(a),size[axis]/2-0.008)))
    return dict(length_m=float(size[l]),width_m=float(size[w]),height_m=float(size[v]),
                inner_half_lengths_m=inner,interior_depth_m=float(floor_depth),
                source='official_local_collision_mesh_shape_only',
                mesh_axes=[l,w,v],horizontal_plane_areas=levels)


def red_mask(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    return (((hsv[:,:,0]<=10)|(hsv[:,:,0]>=170)) & (hsv[:,:,1]>=100) & (hsv[:,:,2]>=45))


def colour_mask(bgr,colour):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);h=hsv[:,:,0]
    ranges={'red':(h<=10)|(h>=170),'yellow':(h>=18)&(h<=38),
            'green':(h>=40)&(h<=90),'blue':(h>=95)&(h<=135)}
    if colour not in ranges: raise GeometryError('UNSUPPORTED_TARGET_COLOUR')
    # New held-object verification mask. Frozen Day 4 thresholds are untouched.
    return ranges[colour] & (hsv[:,:,1]>=90) & (hsv[:,:,2]>=55)


def masked_cloud(mask,depth,rgbK,depthK,T,step=2,depth_to_rgb=None):
    """Select depth points by live RGB mask using actual calibrated extrinsics.

    No PointCloud2 publisher or background full-cloud pipeline is required.
    The temporary vectorized grid is used only during endpoint measurement.
    """
    if depth_to_rgb is not None and not np.allclose(depth_to_rgb,np.eye(4),atol=1e-7):
        vv,uu=np.mgrid[0:depth.shape[0]:step,0:depth.shape[1]:step]
        u=uu.ravel();v=vv.ravel();z=depth[v,u]
        valid=np.isfinite(z)&(z>=.2)&(z<=5.)
        u,v,z=u[valid],v[valid],z[valid]
        dx,dy,dcx,dcy=depthK
        points=np.column_stack(((u-dcx)*z/dx,(v-dcy)*z/dy,z))
        colour=apply(depth_to_rgb,points);front=colour[:,2]>.05
        points=points[front];colour=colour[front]
        fx,fy,cx,cy=rgbK
        cu=np.rint(fx*colour[:,0]/colour[:,2]+cx).astype(int)
        cv=np.rint(fy*colour[:,1]/colour[:,2]+cy).astype(int)
        valid=(cu>=0)&(cv>=0)&(cu<mask.shape[1])&(cv<mask.shape[0])
        points,colour,cu,cv=points[valid],colour[valid],cu[valid],cv[valid]
        # Z-buffer per projected RGB pixel avoids selecting an occluded layer.
        order=np.argsort(colour[:,2]);pixel=cv*mask.shape[1]+cu
        _,first=np.unique(pixel[order],return_index=True);keep=order[first]
        points,cu,cv=points[keep],cu[keep],cv[keep]
        take=mask[cv,cu]
        return apply(T,points[take]),np.column_stack((cu[take],cv[take]))
    v,u=np.nonzero(mask);u=u[::step];v=v[::step]
    fx,fy,cx,cy=rgbK;dx,dy,dcx,dcy=depthK
    du=np.rint((u-cx)/fx*dx+dcx).astype(int)
    dv=np.rint((v-cy)/fy*dy+dcy).astype(int)
    good=(du>=0)&(dv>=0)&(du<depth.shape[1])&(dv<depth.shape[0])
    u,v,du,dv=u[good],v[good],du[good],dv[good]
    z=depth[dv,du];good=np.isfinite(z)&(z>=0.2)&(z<=5.)
    z,du,dv=z[good],du[good],dv[good]
    p=np.column_stack(((du-dcx)*z/dx,(dv-dcy)*z/dy,z))
    return apply(T,p),np.column_stack((u[good],v[good]))


def fit_bin(points,profile,near_xy):
    p=np.asarray(points,dtype=float)
    p=p[np.isfinite(p).all(1)]
    # A sensor-derived Day 7 location gates association; no global coordinates.
    p=p[(np.linalg.norm(p[:,:2]-np.asarray(near_xy),axis=1)<0.72)&(p[:,2]>.20)]
    if len(p)<150: raise GeometryError('TOO_FEW_RED_BIN_DEPTH_POINTS')
    rim_z=float(np.percentile(p[:,2],99))
    rim=p[p[:,2]>=rim_z-.015]
    if len(rim)<45: raise GeometryError('RIM_NOT_OBSERVED')
    rect=cv2.minAreaRect(rim[:,:2].astype(np.float32))
    corners=cv2.boxPoints(rect).astype(float)
    e=corners[1]-corners[0];f=corners[2]-corners[1]
    if np.linalg.norm(e)<np.linalg.norm(f): e,f=f,e
    L,W=np.linalg.norm(e),np.linalg.norm(f)
    if abs(L-profile['length_m'])>.055 or abs(W-profile['width_m'])>.045:
        raise GeometryError('RIM_INCOMPLETE_OR_OCCLUDED: measured L,W='+str((L,W)))
    ex=e/L
    if ex[0]<-1e-6 or (abs(ex[0])<1e-6 and ex[1]<0):ex=-ex
    ey=np.array([-ex[1],ex[0]])
    R=np.eye(3); R[:2,0]=ex;R[:2,1]=ey
    center=np.r_[corners.mean(0),rim_z]
    T=transform(R,center);local=apply(np.linalg.inv(T),rim)
    supports=[]
    for axis,half in ((0,L/2),(1,W/2)):
        for sign in (-1,1):
            supports.append(int(np.count_nonzero(np.abs(local[:,axis]-sign*half)<.022)))
    if min(supports)<7: raise GeometryError('NOT_ALL_RIM_EDGES_SUPPORTED')
    floor_z=rim_z-profile['interior_depth_m']
    local_all=apply(np.linalg.inv(T),p)
    hx,hy=profile['inner_half_lengths_m']
    floor_candidates=p[(np.abs(local_all[:,0])<hx-.025)&(np.abs(local_all[:,1])<hy-.025)
                       &(p[:,2]<rim_z-.07)]
    floor_observed=False
    if len(floor_candidates)>=25:
        low=float(np.percentile(floor_candidates[:,2],12))
        plane=floor_candidates[np.abs(floor_candidates[:,2]-low)<.012]
        if len(plane)>=20:
            measured=float(np.median(plane[:,2]))
            if abs(measured-floor_z)>.025:
                raise GeometryError('VISIBLE_FLOOR_DISAGREES_WITH_OFFICIAL_MESH')
            floor_z=measured;floor_observed=True
    return dict(base_T_bin=T.tolist(),rim_z_m=rim_z,floor_z_m=floor_z,
                table_z_m=rim_z-profile['height_m'],
                outer_half_lengths_m=[L/2,W/2],inner_half_lengths_m=[hx,hy],
                rim_edge_support_counts=supports,red_depth_points=len(p),rim_points=len(rim),
                floor_source=('live_depth_plane' if floor_observed else
                              'live_rim_minus_official_mesh_interior_depth'),
                floor_directly_observed=floor_observed,
                live_geometry=True,bin_mesh_profile=profile)


def book_attachment(model,joints,day5,day4,arm):
    gp=day5['grasp_plan'];candidate=gp['candidate_plans'][arm]
    q=dict(joints)
    q.update(zip(candidate['joint_names'],candidate['grasp']['positions_rad']))
    chain=model.chain('base_footprint',f'gripper_{arm}_grasping_link')
    Tg=model.forward(chain,q).tip_transform
    # Measured exposed spine centre plus half the public 16 cm book depth.
    # Canonical orientation is depth-forward / thin-lateral / height-up.
    surface=np.asarray(day4['target_book_geometry']['base_point_xyz_m'],float)
    Tb=transform(np.eye(3),surface+np.array([.08,0,0]))
    return np.linalg.inv(Tg)@Tb


def validate_book_cloud(points,T_tip,T_tip_book,minimum=20):
    Tb=T_tip@T_tip_book;p=apply(np.linalg.inv(Tb),points)
    h=np.array([.08,.01,.125])
    good=(np.abs(p)<=h+.030).all(1)
    p=p[good]
    if len(p)<minimum: raise GeometryError('HELD_COLOUR_NOT_NEAR_PREDICTED_BOOK')
    # A partial front surface may be visible. Do not infer unseen object pose
    # from a foreground-colour centroid. Keep the grasp-derived rigid transform
    # and enlarge the planned object by the measured surface uncertainty.
    surface_error=np.min(np.abs(np.abs(p)-h),axis=1)
    residual=float(np.percentile(surface_error,80))
    if residual>.025: raise GeometryError('HELD_BOOK_GEOMETRY_INCONSISTENT')
    return dict(consistent_colour_points=int(len(p)),surface_residual_m=residual,
                attachment_source='measured_Day4_spine_plus_Day5_FK_grasp',
                dimensions_depth_width_height_m=[.16,.02,.25],
                uncertainty_margin_m=max(.010,min(.020,residual+.006)))


def book_bounds_in_bin(Ttip,Tattach,scene,margin=0):
    h=np.array([.08,.01,.125])+margin
    p=apply(np.linalg.inv(np.array(scene['base_T_bin']))@Ttip@Tattach,box_corners(-h,h))
    return p.min(0),p.max(0)


def supported_geometry(Ttip,Tattach,scene,floor_tolerance=.025):
    lo,hi=book_bounds_in_bin(Ttip,Tattach,scene,0)
    hx,hy=np.array(scene['inner_half_lengths_m'])-.008
    floor=scene['floor_z_m']-scene['rim_z_m']
    return (lo[0]>-hx and hi[0]<hx and lo[1]>-hy and hi[1]<hy
            and abs(lo[2]-floor)<=floor_tolerance)
