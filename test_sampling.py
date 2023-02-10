import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
import pdf

#parameters grid
minx=-50
maxx=50
nx=100

miny=-50
maxy=50
ny=100

minz=-50
maxz=50
nz=100

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

sf=pdf.sphere(r=30)
sphere_density = sf(xaxis[:,None,None], yaxis[None,:,None], zaxis[None,None,:])
pdf.save_density(sphere_density, 1.0, "sphere.mrc", origin=None)
rs,points,radii=pdf.sample_surface(sphere_density,0.5,0.5,5)

pdf.save_density(rs, 1.0, "points_sphere.mrc", origin=None)

ex=pdf.extrude(sphere_density,0.5,0.5)
pdf.save_density(ex, 1.0, "extrusion_sphere.mrc", origin=None)

rs,points,radii=pdf.sample_extrusion(ex,5)
pdf.save_density(rs, 1.0, "points_extrusion_sphere.mrc", origin=None)


