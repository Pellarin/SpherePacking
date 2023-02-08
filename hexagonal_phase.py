import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
import pdf

#parameters grid
minx=-150
maxx=150
nx=300

miny=-150
maxy=150
ny=300

minz=-150
maxz=150
nz=300

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


cell_dist=60.0
ratio = np.sqrt(3)/2 # cos(60°)

hexa=np.array([(0,0),(1,0),(-1,0),(-1/2,-ratio),(+1/2,-ratio),(-1/2,ratio),(+1/2,ratio)])
hexa*=cell_dist


pdfs=[]
for t in hexa:
    x=t[0]
    y=t[1]
    pdfs.append(pdf.cylinder(x,y,r=20))


# create the cylinders
joint=pdf.joinpdf(pdfs)
# remove cylinders exceeding a distance
#cf=cylinder_filter(0,0,r=180)
#filterpdf=intersectpdf([joint,cf])
# remove part of the cyliner outside the sphere
sf=pdf.sphere_filter(r=140)
filterpdf=pdf.intersectpdf([joint,sf])

cylinders_density = filterpdf(xaxis[:,None,None], yaxis[None,:,None], zaxis[None,None,:])
pdf.save_density(cylinders_density, 1.0, "cylinders.mrc", origin=None)

# add an external sphere
sphere=pdf.sphere(r=140,tolerance=1000)
sphere_density = sphere(xaxis[:,None,None], yaxis[None,:,None], zaxis[None,None,:])
pdf.save_density(sphere_density, 1.0, "sphere.mrc", origin=None)


cylinder_threshold=(0.06,0.01)
sphere_threshold=(0.04,0.01)

rs,points,radii=pdf.sample_surface(cylinders_density,cylinder_threshold[0],cylinder_threshold[1],min_distance_beads=4)
pdf.save_density(rs, 1.0, "points_cylinder.mrc", origin=None)
pickle.dump((points,radii),open("points_cylinder.pkl","wb"))

rs,points,radii=pdf.sample_surface(sphere_density,sphere_threshold[0],sphere_threshold[1],min_distance_beads=6)
pdf.save_density(rs, 1.0, "points_sphere.mrc", origin=None)
pickle.dump((points,radii),open("points_sphere.pkl","wb"))
