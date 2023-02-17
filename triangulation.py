import open3d as o3d
import pickle
import numpy as np
import IMP
import IMP.display
import IMP.rmf
import RMF


coordinates,radii=pickle.load(open("sp_median_skin_points.pkl","rb"))

print(len(coordinates))

pcs=o3d.utility.Vector3dVector(np.array(coordinates))



pc=o3d.geometry.PointCloud(pcs)
print(len(np.asarray(pc.points)))

pc.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8.0, max_nn=1))

'''
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, width=0, scale=1.1, linear_fit=False)[0]
decimated_poisson_mesh=poisson_mesh.simplify_quadric_decimation(1000)
'''

ball_mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc,o3d.utility.DoubleVector(np.array([50.0])))
decimated_mesh=ball_mesh.simplify_quadric_decimation(1000)


#o3d.visualization.draw_geometries([poisson_mesh],mesh_show_wireframe=True)

#print(np.asarray(len(poisson_mesh.vertices)))
#print(np.max(np.asarray(poisson_mesh.triangles)))

vert=np.asarray(decimated_mesh.vertices)
geometries=[]
for tri in np.asarray(decimated_mesh.triangles):

    p0=vert[tri[0]]
    p1=vert[tri[1]]
    p2=vert[tri[2]]
    s3d=IMP.algebra.Segment3D(p0,p1)
    geometries.append(IMP.display.SegmentGeometry(s3d))
    s3d=IMP.algebra.Segment3D(p1,p2)
    geometries.append(IMP.display.SegmentGeometry(s3d))
    s3d=IMP.algebra.Segment3D(p2,p0)
    geometries.append(IMP.display.SegmentGeometry(s3d))
    
rh = RMF.create_rmf_file("sphere.triangulation.rmf")
IMP.rmf.add_geometries(rh, geometries)
IMP.rmf.save_frame(rh)
del rh
