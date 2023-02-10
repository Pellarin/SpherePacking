import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
import pdf




#create an hexagonal mesh 
import numpy as np
import matplotlib.pyplot as plt

g=pdf.Grid((-250,-250,-250),(250,250,250))


sppdf=pdf.SchwarzP(500.0)
sp_density = g.evaluate(sppdf)
pdf.save_density(sp_density, 1.0, "sp.mrc", origin=None)

skin=pdf.get_skin(sp_density,0.27,0.03)
pdf.save_density(skin, 1.0, "sp_median_skin.mrc", origin=None)
radius=20.0
points=pdf.sample_skin(skin,min_distance_beads=radius)
radii=[radius]*len(points)
sampled=pdf.get_sparse_grid_from_points(sp_density,points)
pdf.save_density(sampled, 1.0, "sp_median_skin_points.mrc", origin=None)
pickle.dump((points,radii),open("sp_median_skin_points.pkl","wb"))

'''
# compute the extrusion
extrude_skin=pdf.extrude(skin)
pdf.save_density(extrude_skin, 1.0, "sp_extrude_skin.mrc", origin=None)
radius=20.0
points=pdf.sample_skin(extrude_skin,min_distance_beads=radius)
radii=[radius]*len(points)
sampled=pdf.get_sparse_grid_from_points(sp_density,points)
pdf.save_density(sampled, 1.0, "sp_extrude_skin_points.mrc", origin=None)
pickle.dump((points,radii),open("sp_extrude_skin_points.pkl","wb"))

'''


