import numba
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


def unwrap_coordinates(snap):
    """Unwrap periodic boundary conditions"""

    box = np.asarray(snap.configuration.box[:3], dtype=np.float32)
    positions = np.asarray(snap.particles.position, dtype=np.float32)

    backbone_bonds = snap.bonds.group[snap.bonds.typeid == 0]
    
    _unwrap_backbone(positions, backbone_bonds, box)

    if isinstance(snap.particles.image, np.ndarray):
        chrom_bounds = get_chrom_bounds(snap)

        for bounds in chrom_bounds:
            telomere_image = snap.particles.image[bounds[0]]
            chrom_positions = positions[bounds[0]:bounds[1]+1]

            chrom_positions += (telomere_image*box)[None, :]

    return positions
    
    
@numba.njit("void(f4[:,:], u4[:,:], f4[:])")
def _unwrap_backbone(_positions, _backbone_bonds, _box):
    """Unwrap chromosome backbone(s)"""
    
    for bond in _backbone_bonds:
        p0 = _positions[bond[0]]
        p1 = _positions[bond[1]]
        
        for i in range(3):
            PBC_shift = round((p1[i]-p0[i])/_box[i])
            p1[i] -= _box[i] * PBC_shift
