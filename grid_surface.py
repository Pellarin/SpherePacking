import numpy as np
from scipy import spatial
import mrcfile
import random

#parameters grid
minx=-70
maxx=70
nx=140

miny=-70
maxy=70
ny=140

minz=-70
maxz=70
nz=140

#density selection
density_skin=0.9
density_skin_thickness=0.1

#beads restraints
min_distance_beads=5.0

def save_density(data, grid_spacing, outfilename, origin=None):
    """
    Save the density to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data.T)
        mrc.voxel_size = grid_spacing
        if origin is not None:
            mrc.header['origin']['x'] = origin[0]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("done")



def sphere(x, y, z, tolerance=100, r=30):
    argument=np.abs(x**2+y**2+z**2-r**2)/tolerance
    return np.exp(-argument)

def join(f1,f2):
    return (1.0-(1.0-f1)*(1.0-f2))
    
def intersect(f1,f2):
    return f1*f2
    
def hyperboloid(x,y,z):
    tolerance=40
    argument=np.abs(y**2/100+x**2/100+z**2/100-1)/tolerance
    #argument=np.abs(y**2/100-x**2/100-z)/tolerance
    return np.exp(-argument)

def paraboloid(x,y,z):
    tolerance=40
    argument=np.abs(y**2/100-x**2/100-z)/tolerance
    return np.exp(-argument)

xaxis = np.linspace(minx,maxx, nx)
yaxis = np.linspace(miny,maxy, ny)
zaxis = np.linspace(minz,maxz, nz)

result = sphere(xaxis[:,None,None], yaxis[None,:,None],zaxis[None,None,:])

save_density(result, 1.0, "surface.mrc", origin=None)

w=np.where((result>=density_skin) & (result<density_skin+density_skin_thickness))
wt=list(zip(*w))
rest=np.ones_like(range(len(wt)))
print(rest)
points=[]
tree = spatial.KDTree(wt)
while np.sum(rest)>0:
    print(len(points),np.sum(rest))
    choice_index=np.random.choice(np.where(rest == 1)[0])
    print(choice_index)
    v=wt[choice_index]
    points.append(v)

    indexes=tree.query_ball_point(v,r=min_distance_beads)
    for index in indexes:
        try:
            rest[index]=0
        except:
            continue

rs=np.zeros_like(result)
for v in points:
    rs[v]=1.0
    

save_density(rs, 1.0, "points.mrc", origin=None)
