import IMP
import IMP.atom
import IMP.core
import IMP.display
import random
import pickle
import numpy as np
from scipy.spatial import Delaunay
import math
from matplotlib import colors
from tqdm import tqdm

# create an IMP model
m=IMP.Model()

topologies={"POPC":{"BEADS":[],"BONDS":[],"ANGLES":[]},
            "TEST":{"BEADS":[],"BONDS":[],"ANGLES":[]}}

topologies["TEST"]["BEADS"].append("   1 	Q0 	 1 	POPC 	NC3 	 1 	1.0 	".split())
topologies["TEST"]["BEADS"].append("   2 	Qa 	 1 	POPC 	PO4 	 2 	-1.0 	".split())
topologies["TEST"]["BEADS"].append("   3 	Na 	 1 	POPC 	GL1 	 3 	0 	    ".split())
topologies["TEST"]["BONDS"].append("   1  2 	1 	0.47 	1250 	".split())
topologies["TEST"]["BONDS"].append("   2  3 	1 	0.47 	1250 	".split())
topologies["TEST"]["ANGLES"].append("   1  2  3 	2 	120.0 	25.0 	".split())

#structure of molecule POPC
#; id 	type 	resnr 	residu 	atom 	cgnr 	charge

topologies["POPC"]["BEADS"].append("   1 	Q0 	 1 	POPC 	NC3 	 1 	1.0 	blue".split())
topologies["POPC"]["BEADS"].append("   2 	Qa 	 1 	POPC 	PO4 	 2 	-1.0 	yellow".split())
topologies["POPC"]["BEADS"].append("   3 	Na 	 1 	POPC 	GL1 	 3 	0 	    pink".split())
topologies["POPC"]["BEADS"].append("   4 	Na 	 1 	POPC 	GL2 	 4 	0 	    pink".split())
topologies["POPC"]["BEADS"].append("   5 	C1 	 1 	POPC 	C1A 	 5 	0 	    gray".split())
topologies["POPC"]["BEADS"].append("   6 	C3 	 1 	POPC 	D2A 	 6 	0 	    green".split())
topologies["POPC"]["BEADS"].append("   7 	C1 	 1 	POPC 	C3A 	 7 	0 	    gray".split())
topologies["POPC"]["BEADS"].append("   8 	C1 	 1 	POPC 	C4A 	 8 	0 	    gray".split())
topologies["POPC"]["BEADS"].append("   9 	C1 	 1 	POPC 	C1B 	 9 	0 	    gray".split())
topologies["POPC"]["BEADS"].append("  10 	C1 	 1 	POPC 	C2B 	10 	0 	    gray".split())
topologies["POPC"]["BEADS"].append("  11 	C1 	 1 	POPC 	C3B 	11 	0 	    gray".split())
topologies["POPC"]["BEADS"].append("  12 	C1 	 1 	POPC 	C4B 	12 	0 	    gray".split())
  
#[bonds]
#;  i  j 	funct 	length 	force.c.
topologies["POPC"]["BONDS"].append("   1  2 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   2  3 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   3  4 	1 	0.37 	1250 	".split())
topologies["POPC"]["BONDS"].append("   3  5 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   5  6 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   6  7 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   7  8 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   4  9 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("   9 10 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("  10 11 	1 	0.47 	1250 	".split())
topologies["POPC"]["BONDS"].append("  11 12 	1 	0.47 	1250 	".split())

#[angles]
#;  i  j  k 	funct 	angle 	force.c.
topologies["POPC"]["ANGLES"].append("   2  3  4 	2 	120.0 	25.0 	".split())
topologies["POPC"]["ANGLES"].append("   2  3  5 	2 	180.0 	25.0 	".split())
topologies["POPC"]["ANGLES"].append("   3  5  6 	2 	180.0 	25.0 	".split())
topologies["POPC"]["ANGLES"].append("   5  6  7 	2 	120.0 	45.0 	".split())
topologies["POPC"]["ANGLES"].append("   6  7  8 	2 	180.0 	25.0 	".split())
topologies["POPC"]["ANGLES"].append("   4  9 10 	2 	180.0 	25.0 	".split())
topologies["POPC"]["ANGLES"].append("   9 10 11 	2 	180.0 	25.0 	".split())
topologies["POPC"]["ANGLES"].append("  10 11 12 	2 	180.0 	25.0 	".split())  

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def get_particle(x,y,z,r,is_optimized):
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
    dr.set_coordinates_are_optimized(is_optimized)
    
    # create a hierarchy
    ha=IMP.atom.Hierarchy(pa)

    #now create the movers
    mva=IMP.core.BallMover(m,pa,5)
        
    return ha,mva
    
class Build(object):
    def __init__(self,hroot,topology_data):
        self.hroot=hroot
        self.topology=topology_data
        self.rs=[]
        self.residue_types={}
        self.excluded_particle_pairs=[]
        
    def init_molecule(self,name,x,y,z,r,is_optimized):
        h=IMP.atom.Hierarchy(IMP.Particle(m))
        mol=IMP.atom.Molecule.setup_particle(h)
        mol.set_name(name)
        for pdata in self.topology[name]["BEADS"]:
            if pdata[1] in self.residue_types:
                rt=self.residue_types[pdata[1]]
            else:
                rt=IMP.atom.ResidueType(pdata[1])
            p,mva=get_particle(x,y,z,r,is_optimized)
            res=IMP.atom.Residue.setup_particle(p,rt)
            res.set_name(pdata[4])
            res.set_index(int(pdata[0]))
            IMP.display.Colored.setup_particle(p,IMP.display.Color(*colors.to_rgb(pdata[7])))
            mol.add_child(res)
        self.hroot.add_child(mol) 
        
        for pdata in self.topology[name]["BONDS"]:
            s=IMP.atom.Selection(mol,residue_indexes=[int(pdata[0]),int(pdata[1])])
            p1,p2=s.get_selected_particles()
            if not IMP.atom.Bonded.get_is_setup(p1):
                IMP.atom.Bonded.setup_particle(p1)
            if not IMP.atom.Bonded.get_is_setup(p2):
                IMP.atom.Bonded.setup_particle(p2)

            if not IMP.atom.get_bond(IMP.atom.Bonded(p1), IMP.atom.Bonded(p2)):
                IMP.atom.create_bond(
                    IMP.atom.Bonded(p1),
                    IMP.atom.Bonded(p2), 1)
            
            self.excluded_particle_pairs.append((p1,p2))
            hf = IMP.core.Harmonic(float(pdata[3])*10,1.0)
            dr=IMP.core.DistanceRestraint(m,hf,p1,p2)
            self.rs.append(dr)
            
        for pdata in self.topology[name]["ANGLES"]:
            s=IMP.atom.Selection(mol,residue_indexes=[int(pdata[0]),int(pdata[1]),int(pdata[2])])
            p1,p2,p3=s.get_selected_particles()
            hf = IMP.core.Harmonic(float(pdata[4])*math.pi/180.0,1.0)
            ar=IMP.core.AngleRestraint(m, hf, p1, p2, p3)
            self.rs.append(ar)            
                                      
hroot=IMP.atom.Hierarchy(IMP.Particle(m))

build=Build(hroot,topologies)

hroot_head=IMP.atom.Hierarchy(IMP.Particle(m))
heads,radii=pickle.load(open("sphere_extrude_skin_points.pkl","rb"))
head_particles=[]
for n,v in tqdm(enumerate(heads)):
    build.init_molecule("POPC",*v,3,True)
    h,mv=get_particle(*v,radii[n],False)
    hroot_head.add_child(h)
    head_particles.append(h)
    

hroot_foot=IMP.atom.Hierarchy(IMP.Particle(m))
foot_anchors,radii=pickle.load(open("sphere_median_skin_points.pkl","rb"))
foot_particles=[]
for n,v in tqdm(enumerate(foot_anchors)):
    h,mv=get_particle(*v,radii[n],False)
    hroot_foot.add_child(h)
    foot_particles.append(h)



anchor_restraints=[]
lip_feet=IMP.atom.Selection(hroot,molecule="POPC",residue_indexes=[8,12]).get_selected_particles()
for p in tqdm(lip_feet):
    index=closest_node(IMP.core.XYZ(p).get_coordinates(),foot_anchors)
    pa=foot_particles[index]
    hf = IMP.core.Harmonic(5.0,1.0)
    dr=IMP.core.DistanceRestraint(m,hf,pa,p)
    anchor_restraints.append(dr)
    
lip_head=IMP.atom.Selection(hroot,molecule="POPC",residue_indexes=[2]).get_selected_particles()
for p in tqdm(lip_head):
    index=closest_node(IMP.core.XYZ(p).get_coordinates(),heads)
    pa=head_particles[index]
    hf = IMP.core.Harmonic(0.0,1.0)
    dr=IMP.core.DistanceRestraint(m,hf,pa,p)
    anchor_restraints.append(dr)





sf = IMP.core.RestraintsScoringFunction(build.rs+anchor_restraints)
cg = IMP.core.SteepestDescent(m)
cg.set_scoring_function(sf)

import IMP.rmf
import RMF

rh = RMF.create_rmf_file("POPC.plane.rmf")
IMP.rmf.add_hierarchies(rh, [hroot])


print(0,sf.evaluate(False))
# run the sampling
for i in range(200):
    cg.optimize(10)
    print(i,sf.evaluate(False))

ssps = IMP.core.SoftSpherePairScore(1.0)
lsa = IMP.container.ListSingletonContainer(m)
lsa.add(IMP.get_indexes(IMP.atom.get_leaves(hroot)))
rbcpf = IMP.core.RigidClosePairsFinder()
cpc = IMP.container.ClosePairContainer(lsa, 0.0, rbcpf, 10.0)

inverted = [(p1, p0) for p0, p1 in build.excluded_particle_pairs]
lpc = IMP.container.ListPairContainer(m)
lpc.add(IMP.get_indexes(build.excluded_particle_pairs))
lpc.add(IMP.get_indexes(inverted))
icpf = IMP.container.InContainerPairFilter(lpc)
cpc.add_pair_filter(icpf)


evr = IMP.container.PairsRestraint(ssps, cpc)





sf = IMP.core.RestraintsScoringFunction(build.rs+anchor_restraints+[evr])
cg = IMP.core.ConjugateGradients(m)
cg.set_scoring_function(sf)

print(0,sf.evaluate(False))
# run the sampling
for i in range(50):
    cg.optimize(10)
    print(i,sf.evaluate(False))
    

for r in anchor_restraints: r.set_weight(0)

print(0,sf.evaluate(False))
# run the sampling
for i in range(50):
    cg.optimize(10)
    print(i,sf.evaluate(False))

IMP.rmf.save_frame(rh)
del rh

