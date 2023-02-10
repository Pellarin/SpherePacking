import pickle
import pdf




#create an hexagonal mesh 
import numpy as np
import matplotlib.pyplot as plt

g=pdf.Grid((-150,-150,-150),(150,150,150))


sppdf=pdf.sphere(r=130.0)
sp_density = g.evaluate(sppdf)
pdf.save_density(sp_density, 1.0, "sphere.mrc", origin=None)

skin=pdf.get_skin(sp_density,0.90,0.1)
pdf.save_density(skin, 1.0, "sphere_median_skin.mrc", origin=None)
radius=10.0
points=pdf.sample_skin(skin,min_distance_beads=radius)
radii=[radius/2]*len(points)
sampled=pdf.get_sparse_grid_from_points(sp_density,points)
pdf.save_density(sampled, 1.0, "sphere_median_skin_points.mrc", origin=None)
pickle.dump((points,radii),open("sphere_median_skin_points.pkl","wb"))


# compute the extrusion
extrude_skin=pdf.extrude(skin)
pdf.save_density(extrude_skin, 1.0, "sphere_extrude_skin.mrc", origin=None)
radius=10.0
points=pdf.sample_skin(extrude_skin,min_distance_beads=radius)
radii=[radius/2]*len(points)
sampled=pdf.get_sparse_grid_from_points(sp_density,points)
pdf.save_density(sampled, 1.0, "sphere_extrude_skin_points.mrc", origin=None)
pickle.dump((points,radii),open("sphere_extrude_skin_points.pkl","wb"))




