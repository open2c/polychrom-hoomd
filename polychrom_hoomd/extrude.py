import hoomd
import warnings

import numpy as np
import polychrom_hoomd.utils as utils

try:
    import cupy as cp
except ImportError:
    warnings.warn("Could not load cupy library - disabling local topology updates")


def update_topology(system, bond_list, local=True, thermalize=False):
    """Update topology on either GPU or CPU, based on availability"""

    LEF_typeid = system.state.bond_types.index('LEF')
    LEF_dummy_typeid = system.state.bond_types.index('LEF_dummy')
    
    if bond_list:
        # Discard contiguous loops
        bond_array = np.asarray(bond_list, dtype=np.uint32)
        type_array = np.ones(len(bond_array),  dtype=np.uint32) * LEF_typeid

        redundant_bonds = (bond_array[:, 1] - bond_array[:, 0] < 2)
		
        bond_array[redundant_bonds] = np.asarray([0,1], dtype=np.uint32)
        type_array[redundant_bonds] = LEF_dummy_typeid

    else:
        bond_array = np.empty(0, dtype=np.uint32)
        type_array = np.empty(0, dtype=np.uint32)

    if local:
        try:
            _update_topology_local(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)

        except:
            warnings.warn("Reverting to non-local topology update")
            _update_topology_nonlocal(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)

    else:
        _update_topology_nonlocal(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)

    if thermalize:
        system.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)


def _update_topology_nonlocal(system, bond_array, type_array, type_id, dummy_id):
    """Update topology on the CPU"""
    
    snap = system.state.get_snapshot()
    snap_gsd = utils.get_gsd_snapshot(snap)

    non_LEF_ids = (snap.bonds.typeid != type_id)*(snap.bonds.typeid != dummy_id)
		    
    n_LEF = bond_array.shape[0]
    n_non_LEF = np.count_nonzero(non_LEF_ids)
        
    groups = np.zeros((n_non_LEF+n_LEF, 2), dtype=np.uint32)
    typeids = np.zeros(n_non_LEF+n_LEF, dtype=np.uint32)
    
    groups[:n_non_LEF] = snap.bonds.group[non_LEF_ids]
    typeids[:n_non_LEF] = snap.bonds.typeid[non_LEF_ids]

    if n_LEF:
        groups[n_non_LEF:] = bond_array
        typeids[n_non_LEF:] = type_array

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


def _update_topology_local(system, bond_array, type_array, type_id, dummy_id):
    """Update topology locally on the GPU"""
    		
    with system.state.gpu_local_snapshot as local_snap:
        bond_ids = cp.array(local_snap.bonds.typeid, copy=False)

        LEF_ids = cp.equal(bond_ids, type_id)
        LEF_dummy_ids = cp.equal(bond_ids, dummy_id)

        ids = cp.logical_or(LEF_ids, LEF_dummy_ids)

        if bond_array.shape[0] == type_array.shape[0] == cp.count_nonzero(ids):
            local_snap.bonds.group[ids] = bond_array
            local_snap.bonds.typeid[ids] = type_array

        else:
            warnings.warn("Unable to dynamically resize bond arrays on the GPU")
            raise
