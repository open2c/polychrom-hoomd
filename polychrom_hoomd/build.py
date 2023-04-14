import hoomd
import numpy as np

from polykit.generators.initial_conformations import grow_cubic
    
    
def get_init_snapshot(n_beads, n_rep, density, confine, types=None, **kwargs):
    """Setup simulation box and initial chromatin state"""

    n_tot = n_beads*n_rep
    
    if types:
        assert (types.shape[0] == n_tot)
    
    if confine:
        L = 4 * (3*n_tot/(4*np.pi*density))**(1/3.)
    else:
        L = (n_tot/density) ** (1/3.)
        
    snap = hoomd.data.make_snapshot(N=n_tot,
                                    box=hoomd.data.boxdim(L=L, dimensions=3),
                                    particle_types=['A', 'B'])
                            
    chrom = build_chromosomes(n_tot, L, **kwargs)
    chrom -= chrom.mean(axis=0, keepdims=True)

    for i in range(n_tot):
        snap.particles.position[i] = chrom[i]
        snap.particles.diameter[i] = 1.
        
        if types:
            snap.particles.typeid[i] = types[i]
        
    build_topology(snap, n_beads, n_rep, kwargs)
    
    return snap
                

def build_chromosomes(n_tot, L, mode_init, pad=2, **kwargs):
    """Wrapper for chromosome initial conformation generators"""

    if mode_init == "cubic":
        chrom = grow_cubic(n_tot, int(L - pad))
    else:
        raise NotImplementedError("Unsupported initial configuration mode '%s'" % mode_init)
        
    return chrom.astype(np.float32)
    
    
def build_topology(snap, n_beads, n_rep, split_chrom=False, **kwargs):
    """Build membrane/chromosome topology"""

    snap.bonds.types = ['poly_bonds', 'SMC_bonds']
    snap.angles.types = ['poly_angles']

    ids = np.arange(n_beads*n_rep)
    
    bonds = list(zip(ids[:-1], ids[1:]))
    angles = list(zip(ids[:-2], ids[1:-1], ids[2:]))

    if split_chrom:
        chrom_ends = np.arange(1, n_rep) * n_beads

        for e in chrom_ends[::-1]:
            bonds.pop(e-1)
            
            angles.pop(e-1)
            angles.pop(e-2)

    n_bonds = len(bonds)
    n_angles = len(angles)

    snap.bonds.resize(n_bonds)
    snap.angles.resize(n_angles)

    for k, bond in enumerate(bonds):
        snap.bonds.group[k] = bond
        snap.bonds.typeid[k] = 0

    for l, angle in enumerate(angles):
        snap.angles.group[l] = angle
        snap.angles.typeid[l] = 0
