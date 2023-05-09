import gsd.hoomd
import freud.box

import numpy as np


def get_chrom_bounds(snap):
    """
    Infer chromosome bounds from the snapshot topology
    """

    backbone_bonds = snap.bonds.group[snap.bonds.typeid==0]

    bond_breaks, = np.nonzero(backbone_bonds[1:,0] != backbone_bonds[:-1,1])
    chrom_list = np.split(backbone_bonds, bond_breaks+1)
    
    chrom_bounds = np.zeros((len(chrom_list), 2), dtype=np.int32)
    
    for i, bonds in enumerate(chrom_list):
        chrom_bounds[i] = bonds[0,0], bonds[-1,1]

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
    
    
def get_gsd_snapshot(snap_hoomd):
    """Convert HOOMD snapshots to assignable GSD snapshots"""

    snap_gsd = gsd.hoomd.Frame()

    for attr in snap_gsd.__dict__:
        data_gsd = getattr(snap_gsd, attr)
        
        if hasattr(snap_hoomd, attr):
            data_hoomd = getattr(snap_hoomd, attr)

            if hasattr(data_gsd, '__dict__'):
                for prop in data_gsd.__dict__:
                    if hasattr(data_hoomd, prop):
                        setattr(data_gsd, prop, getattr(data_hoomd, prop))
        
    return snap_gsd


def unwrap_coordinates(snap, exclude_array=None):
    """Unwrap periodic boundary conditions"""

    box = freud.box.Box.from_box(snap.configuration.box)
    
    positions = snap.particles.position.copy()
    
    if isinstance(snap.particles.image, np.ndarray):
        images = snap.particles.image.copy()
        
    else:
        images = np.zeros((snap.particles.N, 3), dtype=np.int32)
    
    if isinstance(exclude_array, np.ndarray):
        assert exclude_array.dtype == 'bool'
    
        images = images[exclude_array]
        positions = positions[exclude_array]

    return box.unwrap(positions, images)
