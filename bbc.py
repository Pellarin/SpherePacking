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




sdpdf=pdf.gyroid(200.0)
sd_density = sdpdf(xaxis[:,None,None], yaxis[None,:,None], zaxis[None,None,:])
pdf.save_density(sd_density, 1.0, "sd.mrc", origin=None)


rs,points,radii=pdf.sample_surface(sd_density,0.6,0.3,min_distance_beads=6)
pdf.save_density(rs, 1.0, "points_sd.mrc", origin=None)
pickle.dump((points,radii),open("points_sd.pkl","wb"))

