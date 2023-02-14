import IMP
import IMP.atom
import IMP.core


# create an IMP model
m=IMP.Model()

def get_particle(x,y,z,r):
    # create a new particle
    pa=IMP.Particle(m)
    dr=IMP.core.XYZR.setup_particle(pa)   
    dr.set_coordinates((x,y,z))
    dr.set_radius(r)
    dr.set_coordinates_are_optimized(True)
    return pa

p1=get_particle(0,0,0,1.0)
p2=get_particle(0.1,0,0,1.0)
hf = IMP.core.Harmonic(1.0,1.0)
dr=IMP.core.DistanceRestraint(m,hf,p1,p2)

sf = IMP.core.RestraintsScoringFunction([dr])
cg = IMP.core.SteepestDescent(m)
cg.set_scoring_function(sf) 

for i in range(100):
    cg.optimize(1)
    print(i,sf.evaluate(True))


