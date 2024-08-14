import hoomd

import numpy as np
import polychrom_hoomd.utils as utils


def update_topology(system, bond_list, thermalize=False):
    """Update topology based on LEF positions"""
    
    snap = system.state.get_snapshot()
    snap_gsd = utils.get_gsd_snapshot(snap)
    
    if bond_list:
		# Discard contiguous loops
		bond_array = np.asarray(bond_list, dtype=np.uint32)
		redundant_bonds = (bond_array[:, 1] - bond_array[:, 0] < 2)
		
		LEF_bonds = bond_array[~redundant_bonds]

		# Discard trans-chromosomal loops
		bond_trans_ids, _ = utils.get_trans_cis_ids(LEF_bonds, snap)
		trans_bonds = (bond_trans_ids[:, 1] != bond_trans_ids[:, 0])
		
		LEF_bonds = LEF_bonds[~trans_bonds]
    
    else:
		LEF_bonds = np.empty(0, dtype=np.uint32)
		    
	LEF_typeid = snap.bonds.types.index('LEF')

    n_LEF = LEF_bonds.shape[0]
    n_non_LEF = np.count_nonzero(snap.bonds.typeid != LEF_typeid)
        
    groups = np.zeros((n_non_LEF+n_LEF, 2), dtype=np.uint32)
    typeids = np.zeros(n_non_LEF+n_LEF, dtype=np.uint32)
    
    groups[:n_non_LEF] = snap.bonds.group[snap.bonds.typeid != LEF_typeid]
    typeids[:n_non_LEF] = snap.bonds.typeid[snap.bonds.typeid != LEF_typeid]

	if n_LEF:
		groups[n_non_LEF:] = LEF_bonds
        typeids[n_non_LEF:] = LEF_typeid

    # Bond resizing in HOOMD v3 requires full array reassignment
    snap_gsd.bonds.N = n_non_LEF + n_LEF

    snap_gsd.bonds.group = groups
    snap_gsd.bonds.typeid = typeids

    # Configuration step/box data requires manual setting as of gsd v2.8
    snap_gsd.configuration.step = system.timestep
    snap_gsd.configuration.box = snap.configuration.box
    
    # Load snapshot and re-thermalize, if required
    snap = hoomd.Snapshot.from_gsd_snapshot(snap_gsd, snap.communicator)
    
    system.state.set_snapshot(snap)
    
    if thermalize:
        system.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
