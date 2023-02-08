import IMP
import IMP.atom
import IMP.core
import random
import dill as pickle
import numpy as np

# create an IMP model
m=IMP.Model()


class InBetweenMover(IMP.core.MonteCarloMover):

    def __init__(self, m, p, ps):
        IMP.core.MonteCarloMover.__init__(self, m, "InBetweenMover %1%")
        self.p=p
        self.ps=ps
        
    def do_propose(self):
        rps=random.sample(self.ps,2)
        d1=IMP.core.XYZ(rps[0])
        d2=IMP.core.XYZ(rps[1])
        newx=0.5*(d1.get_x()+d2.get_x())
        newy=0.5*(d1.get_y()+d2.get_y())
        newz=0.5*(d1.get_z()+d2.get_z())
        d=IMP.core.XYZ(self.p)
        self.old_x=d.get_x()
        self.old_y=d.get_y()
        self.old_z=d.get_z()

        
        d.set_x(newx)
        d.set_y(newy)
        d.set_z(newz)
        return IMP.core.MonteCarloMoverResult([self.p.get_index()],1.0)
    
    def do_reject(self):
        d=IMP.core.XYZ(self.p)
        d.set_x(self.old_x)
        d.set_y(self.old_y)
        d.set_z(self.old_z)
        
    def do_get_inputs(self):
        return self.ps
    

class SurfaceRestraint(IMP.Restraint):
    """Restrain particles within (or outside) a cylinder.
       The cylinder is aligned along the z-axis and with center x=y=0.
       Optionally, one can restrain the cylindrical angle
    """
    import math

    def __init__(self, m, ps,func, radius1=75, radius2=45, softness=100.0):
        '''
        @param objects PMI2 objects to restrain
        @param resolution the resolution you want the restraint to be applied
        @param radius the radius of the cylinder
        @param mintheta minimum cylindrical angle in degrees
        @param maxtheta maximum cylindrical angle in degrees
        @param repulsive If True, restrain the particles to be outside
               of the cylinder instead of inside
        @param label A unique label to be used in outputs and
               particle/restraint names
        '''
        IMP.Restraint.__init__(self, m, "SurfaceRestraint %1%")
        self.radius1 = radius1
        self.radius2 = radius2        
        self.softness = softness
        self.particles = [p.get_particle() for p in ps] 
        self.weigth=1
        self.func=func

    def get_score(self, p):
        xyz = IMP.core.XYZ(p)
        jitter=0.00001
        prob = self.func(xyz.get_x(),xyz.get_y(),xyz.get_z())
        score=-self.math.log(prob)
        return score


    def unprotected_evaluate(self, da):
        s = 0.0
        for p in self.particles:
            s += self.get_score(p)
        return s

    def do_get_inputs(self):
        return self.particles






def get_particle(x,y,z,r):
    # create a new particle
    pa=IMP.Particle(m)

    # set the name
    pa.set_name("My Particle A")

    # decorate it as a sphere
    dr=IMP.core.XYZR.setup_particle(pa)

    # set the coordinates     
    dr.set_coordinates((x,y,z))

    # set the radius
    dr.set_radius(r)

    # set the mass
    IMP.atom.Mass.setup_particle(pa,100.0)

    # set the optimization of the coordinates to True
    dr.set_coordinates_are_optimized(True)
    
    # create a hierarchy
    ha=IMP.atom.Hierarchy(pa)

    #now create the movers
    mva=IMP.core.BallMover(m,pa,5)
    return ha,mva
    
mvs=[]
hroot1=IMP.atom.Hierarchy(IMP.Particle(m))
hroot2=IMP.atom.Hierarchy(IMP.Particle(m))

points,radii=pickle.load(open("points_sphere.pkl","rb"))

for n,v in enumerate(points):
    h,mva=get_particle(*v,radii[n])
    mvs.append(mva)
    hroot1.add_child(h)
    
print(len(points))

points,radii=pickle.load(open("points_cylinder.pkl","rb"))

for n,v in enumerate(points):
    h,mva=get_particle(*v,radii[n])
    mvs.append(mva)
    hroot2.add_child(h)
   
print(len(points))


'''
ssps = IMP.core.SoftSpherePairScore(1.0)
lsa = IMP.container.ListSingletonContainer(m)
lsa.add(IMP.get_indexes(IMP.atom.get_leaves(hroot)))
rbcpf = IMP.core.RigidClosePairsFinder()
cpc = IMP.container.ClosePairContainer(lsa, 0.0, rbcpf, 10.0)
evr = IMP.container.PairsRestraint(ssps, cpc)
sr=SurfaceRestraint(m,IMP.atom.get_leaves(hroot),func)
# wrap the restraints in a Scoring Function
sf = IMP.core.RestraintsScoringFunction([evr,sr])

# Build the Monte Carlo Sampler
mc = IMP.core.MonteCarlo(m)
mc.set_scoring_function(sf)
sm = IMP.core.SerialMover(mvs)
mc.add_mover(sm)
mc.set_return_best(False)
mc.set_kt(1.0)
'''


# Prepare the trajectory file
import IMP.rmf
import RMF

rh = RMF.create_rmf_file("out.rmf")
IMP.rmf.add_hierarchies(rh, [hroot1,hroot2])
IMP.rmf.save_frame(rh)


exit()


# run the sampling
for i in range(100):
    print(i)
    mc.optimize(1000)
    IMP.rmf.save_frame(rh)
    print(sf.evaluate(False))
    
del rh


