import numpy as np
from scipy import spatial
import mrcfile
import random
import dill as pickle
from scipy.spatial import distance
from tqdm import tqdm
import sys






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

def extrude(result,density_skin,density_skin_thickness, extrusion1=10, extrusion2=13):
    w=np.where((result>=density_skin) & (result<density_skin+density_skin_thickness))
    wt=list(zip(*w))
    rest1=np.zeros_like(result)
    for p in tqdm(wt):
        rest1[p[0]-extrusion1:p[0]+extrusion1,p[1]-extrusion1:p[1]+extrusion1,p[2]-extrusion1:p[2]+extrusion1]=1.0
    rest2=np.zeros_like(result)
    
    for p in tqdm(wt):
        rest2[p[0]-extrusion2:p[0]+extrusion2,p[1]-extrusion2:p[1]+extrusion2,p[2]-extrusion2:p[2]+extrusion2]=1.0
    
    rest=rest2-rest1
    return rest

def sample_extrusion(extrusion,min_distance_beads=10.0):
    w=np.where(extrusion==1)
    wt=list(zip(*w))
    rest=np.ones_like(range(len(wt)))
    points=[]
    tree = spatial.KDTree(wt)

    total=np.sum(rest)
    while np.sum(rest)>0:
        #progress bar
        sys.stdout.write('\r')
        sys.stdout.write(str(np.sum(rest)/total))
        sys.stdout.flush()
        choice_index=np.random.choice(np.where(rest == 1)[0])
        v=wt[choice_index]
        points.append(v)

        indexes=tree.query_ball_point(v,r=min_distance_beads)
        for i in indexes: rest[i]=0


    rs=np.zeros_like(extrusion)
    for p in points:
        rs[p]=1.0
    
    
    radii=[min_distance_beads/2]*len(points)
    return rs,points,radii

def sample_surface(result,density_skin,density_skin_thickness,min_distance_beads=4.0):
    w=np.where((result>=density_skin) & (result<density_skin+density_skin_thickness))
    wt=list(zip(*w))
    rest=np.ones_like(range(len(wt)))
    points=[]
    tree = spatial.KDTree(wt)
    
    total=np.sum(rest)
    while np.sum(rest)>0:
        #progress bar
        sys.stdout.write('\r')
        sys.stdout.write(str(np.sum(rest)/total))
        sys.stdout.flush()
    
        choice_index=np.random.choice(np.where(rest == 1)[0])
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
    
    radii=[min_distance_beads/2]*len(points)
    return rs,points,radii

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

