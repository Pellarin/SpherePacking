import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
import pdf

#parameters grid
minx=-350
maxx=350
nx=700

miny=-200
maxy=200
ny=400

minz=-200
maxz=200
nz=400

#density selection
density_skin=0.41
density_skin_thickness=0.01

#beads restraints



xaxis = np.linspace(minx,maxx, nx)
yaxis = np.linspace(miny,maxy, ny)
zaxis = np.linspace(minz,maxz, nz)


#create an hexagonal mesh 
import numpy as np
import matplotlib.pyplot as plt


sphere1=pdf.sphere(xc=140,yc=0,zc=0,r=140,tolerance=1000)
sphere2=pdf.sphere(xc=-100,yc=0,zc=0,r=120,tolerance=1000)

# create the cylinders
joint=pdf.joinpdf([sphere1,sphere2])

sphere_density = joint(xaxis[:,None,None], yaxis[None,:,None], zaxis[None,None,:])
pdf.save_density(sphere_density, 1.0, "sphere_join.mrc", origin=None)

intersect=pdf.intersectpdf([sphere1,sphere2])

sphere_intersect = intersect(xaxis[:,None,None], yaxis[None,:,None], zaxis[None,None,:])
pdf.save_density(sphere_density, 1.0, "sphere_intersect.mrc", origin=None)

remove=sphere_density-sphere_intersect
pdf.save_density(remove, 1.0, "sphere_remove.mrc", origin=None)

extrude=pdf.extrude(remove,0.72,0.5)
pdf.save_density(extrude, 1.0, "sphere_intersect_extrusion.mrc", origin=None)



exit()
cylinder_threshold=(0.06,0.01)
sphere_threshold=(0.04,0.01)

rs,points,radii=pdf.sample_surface(cylinders_density,cylinder_threshold[0],cylinder_threshold[1],min_distance_beads=4)
pdf.save_density(rs, 1.0, "points_cylinder.mrc", origin=None)
pickle.dump((points,radii),open("points_cylinder.pkl","wb"))

rs,points,radii=pdf.sample_surface(sphere_density,sphere_threshold[0],sphere_threshold[1],min_distance_beads=6)
pdf.save_density(rs, 1.0, "points_sphere.mrc", origin=None)
pickle.dump((points,radii),open("points_sphere.pkl","wb"))
