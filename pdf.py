import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
from scipy.spatial import distance
from scipy import ndimage
from tqdm import tqdm
import open3d as o3d
import sys


def save_density(data, grid_spacing, outfilename, origin=None):
    """
    Save the density of a grid to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file "+outfilename)
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

class Grid(object):
    """
    This class created a 3D grid with specified dimensions
    """
    def __init__(self,vector1,vector2,resolution=1):
        #parameters grid
        minx=vector1[0]
        maxx=vector2[0]
        nx=int(np.abs(maxx-minx)/resolution)

        miny=vector1[1]
        maxy=vector2[1]
        ny=int(np.abs(maxy-miny)/resolution)

        minz=vector1[2]
        maxz=vector2[2]
        nz=int(np.abs(maxz-minz)/resolution)

        self.xaxis = np.linspace(minx,maxx, nx)
        self.yaxis = np.linspace(miny,maxy, ny)
        self.zaxis = np.linspace(minz,maxz, nz)
        
    def evaluate(self,pdf):
        # evaluate the grid on a pdf function
        return pdf(self.xaxis[:,None,None], self.yaxis[None,:,None], self.zaxis[None,None,:])




def extrude(skin_as_grid,extrusion1=20, extrusion2=23):
    """
    Extrude a skin with two radii a subtract the two densities to obtain a bilayer skin
    """
    print("Extrusion ")
    w=np.where(skin_as_grid==1.0)
    wt=list(zip(*w))
    
    rest1=np.zeros_like(skin_as_grid)
    for p in tqdm(wt):
        rest1[p[0]-extrusion1:p[0]+extrusion1,p[1]-extrusion1:p[1]+extrusion1,p[2]-extrusion1:p[2]+extrusion1]=1.0
    rest2=np.zeros_like(skin_as_grid)
    for p in tqdm(wt):
        rest2[p[0]-extrusion2:p[0]+extrusion2,p[1]-extrusion2:p[1]+extrusion2,p[2]-extrusion2:p[2]+extrusion2]=1.0
    rest=rest2-rest1
    return rest
    
def dilation_difference(skin_as_grid,inner_niter=20, outer_niter=23):
    """
    Extrude a skin with two radii a subtract the two densities to obtain a bilayer skin
    """
    print("Dilate")
    d_inner=ndimage.binary_dilation(skin_as_grid,iterations=inner_niter)
    d_outer=ndimage.binary_dilation(skin_as_grid,iterations=outer_niter)
    rest=d_outer.astype(float)-d_inner.astype(float)
    return rest
           
def get_skin(evaluated_pdf,density_skin_threshold,density_skin_thickness):
    """
    Get a skin from an evaluated grid
    """
    w=np.where((evaluated_pdf>=density_skin_threshold) & (evaluated_pdf<density_skin_threshold+density_skin_thickness))
    wt=list(zip(*w))
    
    rs=get_sparse_grid_from_points(evaluated_pdf,wt)
    return rs

def get_sparse_grid_from_points(evaluated_pdf,points):
    """
    Return a grid from a list of points
    """
    rs=np.zeros_like(evaluated_pdf)
    for p in points:
        rs[p]=1.0
    return rs

def sample_skin(skin,min_distance_beads=10.0,tolerance=0.001):
    """
    Sample a skin using equidistal points
    """
    print("Sampling a skin ")
    w=np.where(skin==1.0)
    wt=list(zip(*w))
    rest=np.ones_like(range(len(wt)))
    points=[]
    tree = spatial.KDTree(wt)

    total=np.sum(rest)
    sys.stdout.write('Sampling the skin '+str(total)+' grid points \n')
    while np.sum(rest)/total>tolerance:
        #progress bar
        sys.stdout.write('\r')
        sys.stdout.write(str(round(np.sum(rest)/total,7))+" %")
        sys.stdout.flush()
        
        choice_index=np.random.choice(np.where(rest == 1)[0])
        v=wt[choice_index]
        points.append(v)

        indexes=tree.query_ball_point(v,r=min_distance_beads)
        for i in indexes: rest[i]=0

    return points

def triangulate(points):
    """
    Triangulate a list points. Return a list of vertices and triangles defined as a list of indexes of vertices
    """
    pcs=o3d.utility.Vector3dVector(np.array(coordinates))
    pc=o3d.geometry.PointCloud(pcs)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8.0, max_nn=1))
    pc.orient_normals_consistent_tangent_plane(100)
    ball_mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc,o3d.utility.DoubleVector(np.array([20.0])))
    decimated_mesh=ball_mesh.simplify_quadric_decimation(int(len(ball_mesh.triangles)/5))
    decimated_mesh.remove_degenerate_triangles()
    decimated_mesh.remove_duplicated_triangles()
    decimated_mesh.remove_duplicated_vertices()
    decimated_mesh.remove_non_manifold_edges()
    
    vertices=np.asarray(decimated_mesh.vertices)
    triangles=np.asarray(decimated_mesh.triangles)
    
    return vertices, triangles


def planexy(tolerance=100):
    def pdf(x,y,z):
        argument=np.abs((x-x)**2+(y-y)**2+(z)**2)/tolerance
        return np.exp(-argument)
    return pdf

def cylinderz(xcenter,ycenter,r=10.0,tolerance=100):
    def pdf(x,y,z):
        argument=np.abs((x-xcenter)**2+(y-ycenter)**2+(z-z)**2-r**2)/tolerance
        return np.exp(-argument)
    return pdf

def cylinderx(ycenter,zcenter,r=10.0,tolerance=100):
    def pdf(x,y,z):
        argument=np.abs((x-x)**2+(y-ycenter)**2+(z-zcenter)**2-r**2)/tolerance
        return np.exp(-argument)
    return pdf

def cylindery(xcenter,zcenter,r=10.0,tolerance=100):
    def pdf(x,y,z):
        argument=np.abs((x-xcenter)**2+(y-y)**2+(z-zcenter)**2-r**2)/tolerance
        return np.exp(-argument)
    return pdf

def gyroid(period):

    n = 2 * np.pi / period
    def pdf(x,y,z):
        a = np.sin(n*x)*np.cos(n*y)
        b = np.sin(n*y)*np.cos(n*z)
        c = np.sin(n*z)*np.cos(n*x)
        return a + b + c
    return pdf


def SchwarzD(period):
    """
    :param x: a vector of coordinates (x1, x2, x3)
    :param period: length of one period
    :return: An approximation of the Schwarz D "Diamond" infinite periodic minimal surface
    """  
    n = 2*np.pi / period  # might be just pi / period
    def pdf(x,y,z):
        a = np.sin(n*x)*np.sin(n*y)*np.sin(n*z)
        b = np.sin(n*x)*np.cos(n*y)*np.cos(n*z)
        c = np.cos(n*x)*np.sin(n*y)*np.cos(n*z)
        d = np.cos(n*x)*np.cos(n*y)*np.sin(n*z)
        return a + b + c + d
    return pdf


def SchwarzP(period):
    """
    :param x: a vector of coordinates (x1, x2, x3)
    :param period: length of one period
    :return: An approximation of the Schwarz D "Diamond" infinite periodic minimal surface
    """
    n = 2*np.pi / period  # might be just pi / period
    def pdf(x,y,z):
        a = np.cos(n*x)+np.cos(n*y)+np.cos(n*z)
        return a
    return pdf

def sphere(xc=0,yc=0,zc=0,tolerance=100, r=50):
    def pdf(x,y,z):
        argument=np.abs((x-xc)**2+(y-yc)**2+(z-zc)**2-r**2)/tolerance
        return np.exp(-argument)
    return pdf
    
def sphere_filter(xc=0,yc=0,zc=0,tolerance=100, r=50,filterout_external=True):
    def pdf(x,y,z):
        
        if filterout_external:
        
            d=(x-xc)**2+(y-yc)**2+(z-zc)**2-r**2      
            argument=np.where(d>=0,np.abs(d),0.0)/tolerance

        else:
        
            d=(x-xc)**2+(y-yc)**2+(z-zc)**2-r**2
            argument=np.where(d<=0,np.abs(d),0.0)/tolerance        

        return np.exp(-argument)
    return pdf    

def cylinder_filter(xc,yc,r=50.0,tolerance=100,filterout_external=True):
        
    def pdf(x,y,z):
        
        if filterout_external:
        
            d=(x-xc)**2+(y-yc)**2+(z-z)**2-r**2      
            argument=np.where(d>=0,np.abs(d),0.0)/tolerance

        else:
        
            d=(x-xc)**2+(y-yc)**2+(z-z)**2-r**2
            argument=np.where(d<=0,np.abs(d),0.0)/tolerance        

        return np.exp(-argument)
    return pdf

def extrudepdf(evaluated_pdf,threshold,thickness,extrusion=10,tolerance=100):
    # very heavy and slow
    w=np.where((evaluated_pdf>=threshold) & (evaluated_pdf<threshold+thickness))
    point_cloud_target=np.array(list(zip(*w)))
    
    rest=np.zeros_like(evaluated_pdf)
    
    wt=np.where(rest==0)
    point_cloud=list(zip(*wt))
    
    for p in tqdm(point_cloud):
        d=np.min(distance.cdist([p],point_cloud_target))
        argument=(d-extrusion)**2/tolerance
        rest[p]=np.exp(-argument)
    return

def joinpdf(pdfs):
    def pdf(x,y,z):
        accum=1.0-pdfs[0](x,y,z)
        for pdf in pdfs[1:]: accum=accum*(1.0-pdf(x,y,z))
        return 1.0-accum
    return pdf
    
def intersectpdf(pdfs):
    def pdf(x,y,z):
        accum=pdfs[0](x,y,z)
        for pdf in pdfs[1:]: accum=accum*pdf(x,y,z)
        return accum
    return pdf
    
def hyperboloid(x,y,z):
    tolerance=40
    argument=np.abs(y**2/100+x**2/100+z**2/100-1)/tolerance
    #argument=np.abs(y**2/100-x**2/100-z)/tolerance
    return np.exp(-argument)

def paraboloid(x,y,z):
    tolerance=40
    argument=np.abs(y**2/100-x**2/100-z)/tolerance
    return np.exp(-argument)



