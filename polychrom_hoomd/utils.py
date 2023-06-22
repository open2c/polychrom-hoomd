import gsd.hoomd
import numpy as np


def get_chrom_bounds(snap):
    """
    Infer chromosome bounds from the snapshot topology
    """

    backbone_bonds = snap.bonds.group[snap.bonds.typeid == 0]

    bond_breaks, = np.nonzero(backbone_bonds[1:, 0] != backbone_bonds[:-1, 1])
    chrom_list = np.split(backbone_bonds, bond_breaks + 1)
    
    chrom_bounds = np.zeros((len(chrom_list), 2), dtype=np.int32)
    
    for i, bonds in enumerate(chrom_list):
        chrom_bounds[i] = bonds[0, 0], bonds[-1, 1]

    return chrom_bounds


def get_trans_cis_ids(ids, snap):
    """
    Get chromosome/intrachromosomal indices for any collection of monomer (absolute) indices
    """

    chrom_bounds = get_chrom_bounds(snap)
    chrom_ends = np.cumsum(np.diff(chrom_bounds, axis=1) + 1)

    trans_ids = np.digitize(ids, chrom_ends)
    cis_ids = np.mod(ids, chrom_ends[trans_ids-1])
    
    return trans_ids, cis_ids
    
    
def get_gsd_snapshot(snap):
    """Convert HOOMD snapshots to assignable GSD snapshots"""

    snap_gsd = gsd.hoomd.Frame()

    for attr in snap_gsd.__dict__:
        data_gsd = getattr(snap_gsd, attr)
        
        if hasattr(snap, attr):
            data = getattr(snap, attr)

            if hasattr(data_gsd, '__dict__'):
                for prop in data_gsd.__dict__:
                    if hasattr(data, prop):
                        setattr(data_gsd, prop, getattr(data, prop))
        
    return snap_gsd


def unwrap_coordinates(snap, max_delta=1):
    """Unwrap periodic boundary conditions"""

    box = snap.configuration.box[None, :3]
    positions = snap.particles.position.copy()
    
    chrom_bounds = get_chrom_bounds(snap)

    for bounds in chrom_bounds:
        chrom_positions = positions[bounds[0]:bounds[1]+1]

        if isinstance(snap.particles.image, np.ndarray):
            telomere_image = snap.particles.image[bounds[0]]
            chrom_positions += telomere_image*box
            
        for delta in range(1, max_delta+1):
            bond_vectors = chrom_positions[delta:] - chrom_positions[:-delta]
            PBC_shifts = np.round(bond_vectors / box)
            
            chrom_positions[delta:] -= np.cumsum(PBC_shifts, axis=0) * box

    return positions
