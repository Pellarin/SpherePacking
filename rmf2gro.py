import IMP
import IMP.rmf
import RMF
import IMP.pmi
import IMP.pmi.output


m=IMP.Model()
h=IMP.pmi.output.RMFHierarchyHandler(m,"POPC.sphere.rmf")


names=["NC3","PO4","GL1","GL2","C1A","D2A","C3A","C4A","C1B","C2B","C3B","C4B"]


print("MD of POPC membrane, t= 0.0")
print(len(IMP.atom.get_leaves(h)))
atomn=0
for n,mol in enumerate(h.get_children()):
    resn=n+1
    for atom in mol.get_children():
        atomn+=1
        atomn=atomn % 100000
        x,y,z=IMP.core.XYZ(atom).get_coordinates()
        atom_index=int(IMP.atom.Residue(atom).get_name())
        print("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f" % (resn,"POPC",names[atom_index-1],atomn,x/10,y/10,z/10,0,0,0))      
print(50.0, 50.0, 50.0)
