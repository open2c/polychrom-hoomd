import os
import hoomd
import warnings

import numpy as np
import polychrom_hoomd.utils as utils

try:
    import cupy as cp
    dpath = os.path.dirname(os.path.abspath(__file__))
    
    with open(f'{dpath}/kernels/lef_neighbor_search.cuh', 'r') as cuda_file:
        cuda_code = cuda_file.read()
        cuda_module = cp.RawModule(code=cuda_code)
    
    _single_leg_search = cuda_module.get_function('_single_leg_search')
    
except ImportError:
    warnings.warn("Could not load CuPy library - local topology updates unavailable")


def update_topology(system, bond_list, local=True, thermalize=False):
    """Update topology on either GPU or CPU, based on availability"""

    LEF_typeid = system.state.bond_types.index('LEF')
    LEF_dummy_typeid = system.state.bond_types.index('LEF_dummy')
    
    if len(bond_list) > 0:
        # Discard contiguous loops
        bond_array = np.asarray(bond_list, dtype=np.uint32)
        type_array = np.ones(len(bond_array),  dtype=np.uint32) * LEF_typeid

        redundant_bonds = (bond_array[:, 1] - bond_array[:, 0] < 1)
		
        bond_array[redundant_bonds] = np.asarray([0, 1], dtype=np.uint32)
        type_array[redundant_bonds] = LEF_dummy_typeid

    else:
        bond_array = np.empty(0, dtype=np.uint32)
        type_array = np.empty(0, dtype=np.uint32)

    if local:
        try:
            _update_topology_local(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)

        except Exception as e:
            warnings.warn("%s - reverting to non-local topology update" % e)
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
            raise RuntimeError("Unable to dynamically resize bond arrays on the GPU")


def update_topology_3D(system, neighbor_list, leg_off_rate, threads_per_block=256):
    """Attempt stochastic 3D cohesin moves on the GPU"""

    LEF_typeid = system.state.bond_types.index('LEF')

    with system.state.gpu_local_snapshot as local_snap:
        bond_ids = cp.array(local_snap.bonds.typeid, copy=False)
        is_bound = cp.equal(bond_ids, LEF_typeid)
        
        N = int(bond_ids.size)
        N_bound = int(cp.count_nonzero(is_bound))
        
        rng_left = cp.random.random(N_bound, dtype=np.float32)
        rng_right = cp.random.random(N_bound, dtype=np.float32)
        
        unbind_left = cp.less(rng_left, leg_off_rate)
        unbind_right = cp.less(rng_right, leg_off_rate)
        
        unbind = cp.logical_and(unbind_left, unbind_right)

        unbind_left = cp.logical_and(unbind_left, cp.logical_not(unbind))
        unbind_right = cp.logical_and(unbind_right, cp.logical_not(unbind))
        
        anchors = cp.zeros(N, dtype=cp.int32)
        rng = cp.random.random(N, dtype=np.float32)
        
        anchors[is_bound] = cp.where(unbind_right, 1, anchors[is_bound])
        anchors[is_bound] = cp.where(unbind_left, -1, anchors[is_bound])

        groups = local_snap.bonds.group._coerce_to_ndarray()
        tags = local_snap.particles.tag._coerce_to_ndarray()
        rtags = local_snap.particles.rtag._coerce_to_ndarray()
        
        with neighbor_list.gpu_local_nlist_arrays as data:
            nlist = data.nlist._coerce_to_ndarray()
            n_neigh = data.n_neigh._coerce_to_ndarray()
            head_list = data.head_list._coerce_to_ndarray()
            
            num_blocks = (N+threads_per_block-1) // threads_per_block
            
            _single_leg_search(
                (num_blocks,),
                (threads_per_block,),
                (N, nlist, n_neigh, head_list, tags, rtags, anchors, rng, groups)
            )
