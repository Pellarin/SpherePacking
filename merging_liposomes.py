import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
import pdf
import numpy as np
import matplotlib.pyplot as plt

g=pdf.Grid((-350,-200,-200),(350,200,200))
sphere1=pdf.sphere(xc=140,yc=0,zc=0,r=140,tolerance=1000)
sphere2=pdf.sphere(xc=-100,yc=0,zc=0,r=120,tolerance=1000)

# create the cylinders
joint=pdf.joinpdf([sphere1,sphere2])

sphere_density = g.evaluate(joint)
pdf.save_density(sphere_density, 1.0, "ml_join.mrc", origin=None)

sf=pdf.sphere_filter(xc=0,yc=0,zc=0,tolerance=100,r=60,filterout_external=False)
sphere_filter = g.evaluate(sf)
pdf.save_density(sphere_filter, 1.0, "ml_filter.mrc", origin=None)

remove=sphere_density*sphere_filter
pdf.save_density(remove, 1.0, "ml_remove.mrc", origin=None)

skin=pdf.get_skin(remove,0.50,0.02)
extrude=pdf.dilation_difference(skin)
pdf.save_density(extrude, 1.0, "ml_intersect_extrusion.mrc", origin=None)

radius=15.0
points=pdf.sample_skin(skin,min_distance_beads=radius)
radii=[radius/2]*len(points)
sampled=pdf.get_sparse_grid_from_points(extrude,points)
pdf.save_density(sampled, 1.0, "ml_median_skin_points.mrc", origin=None)
pickle.dump((points,radii),open("ml_median_skin_points.pkl","wb"))


radius=15.0
points=pdf.sample_skin(extrude,min_distance_beads=radius)
radii=[radius/2]*len(points)
sampled=pdf.get_sparse_grid_from_points(extrude,points)
pdf.save_density(sampled, 1.0, "ml_extrude_skin_points.mrc", origin=None)
pickle.dump((points,radii),open("ml_extrude_skin_points.pkl","wb"))



