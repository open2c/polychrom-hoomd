import os
import hoomd
import warnings

import polychrom_hoomd.utils as utils

try:
    import cupy as xp
    dpath = os.path.dirname(os.path.abspath(__file__))
        
    with open(f'{dpath}/kernels/lef_neighbor_search.cuh', 'r') as cuda_file:
        cuda_code = cuda_file.read()
        cuda_module = xp.RawModule(code=cuda_code)
    
    _single_leg_search = cuda_module.get_function('_single_leg_search')
    
except ImportError:
    import numpy as xp
    warnings.warn("Could not load CuPy library - local/3D topology updates unavailable")


def update_topology(system, bond_list, local=True, thermalize=False):
    """Update topology on either GPU or CPU, based on availability"""

    LEF_typeid = system.state.bond_types.index('LEF')
    LEF_dummy_typeid = system.state.bond_types.index('LEF_dummy')
    
    if len(bond_list) > 0:
        # Discard contiguous loops
        bond_array = xp.asarray(bond_list, dtype=xp.int32)
        type_array = xp.full(len(bond_array), LEF_typeid, dtype=xp.int32)

        redundant_bonds = xp.less(bond_array[:, 1] - bond_array[:, 0], 1)
        n_prune = int(xp.count_nonzero(redundant_bonds))
        
        ids = xp.random.randint(low=0, high=system.state.N_particles-1, size=n_prune, dtype=xp.int32)
		
        bond_array[redundant_bonds] = xp.stack((ids, ids+1)).T
        type_array[redundant_bonds] = LEF_dummy_typeid

    else:
        bond_array = xp.empty(0, dtype=xp.int32)
        type_array = xp.empty(0, dtype=xp.int32)

    if local:
        try:
            _update_topology_local(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)
            warnings.warn("Using local topology update")

        except Exception as e:
            _update_topology_nonlocal(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)
            warnings.warn("%s - reverting to non-local topology update" % e)

    else:
        _update_topology_nonlocal(system, bond_array, type_array, LEF_typeid, LEF_dummy_typeid)

    if thermalize:
        system.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)


def _update_topology_nonlocal(system, bond_array, type_array, type_id, dummy_id):
    """Update topology on the CPU"""
    
    snap = system.state.get_snapshot()
    snap_gsd = utils.get_gsd_snapshot(snap)

    non_LEF_ids = xp.asarray((snap.bonds.typeid != type_id)*(snap.bonds.typeid != dummy_id))
		    
    n_LEF = bond_array.shape[0]
    n_non_LEF = int(xp.count_nonzero(xp.asarray(non_LEF_ids)))
        
    groups = xp.zeros((n_non_LEF+n_LEF, 2), dtype=xp.uint32)
    typeids = xp.zeros(n_non_LEF+n_LEF, dtype=xp.uint32)
    
    group_array = xp.asarray(snap.bonds.group, dtype=xp.uint32)
    typeid_array = xp.asarray(snap.bonds.typeid, dtype=xp.uint32)

    groups[:n_non_LEF] = group_array[non_LEF_ids]
    typeids[:n_non_LEF] = typeid_array[non_LEF_ids]

    if n_LEF:
        groups[n_non_LEF:] = xp.asarray(bond_array).astype(xp.uint32)
        typeids[n_non_LEF:] = xp.asarray(type_array).astype(xp.uint32)

    # Bond resizing in HOOMD v3 requires full array reassignment
    snap_gsd.bonds.N = n_non_LEF + n_LEF

    snap_gsd.bonds.group = groups.get() if xp.__name__ == 'cupy' else groups
    snap_gsd.bonds.typeid = typeids.get() if xp.__name__ == 'cupy' else typeids

    # Configuration step/box data requires manual setting as of gsd v2.8
    snap_gsd.configuration.step = system.timestep
    snap_gsd.configuration.box = snap.configuration.box
    
    # Load snapshot and re-thermalize, if required
    snap = hoomd.Snapshot.from_gsd_snapshot(snap_gsd, snap.communicator)
    
    system.state.set_snapshot(snap)


def _update_topology_local(system, bond_array, type_array, type_id, dummy_id):
    """Update topology locally on the GPU"""
    		
    with system.state.gpu_local_snapshot as local_snap:
        bond_ids = xp.asarray(local_snap.bonds.typeid)

        is_bound = xp.equal(bond_ids, type_id)
        is_unbound = xp.equal(bond_ids, dummy_id)

        is_LEF = xp.logical_or(is_bound, is_unbound)

        if bond_array.shape[0] == type_array.shape[0] == xp.count_nonzero(is_LEF):
            local_snap.bonds.group[is_LEF] = bond_array.astype(xp.uint32)
            local_snap.bonds.typeid[is_LEF] = type_array.astype(xp.uint32)

        else:
            raise RuntimeError("Unable to dynamically resize bond arrays on the GPU")


def update_topology_3D(system, neighbor_list, leg_off_rate, cutoff, threads_per_block=256):
    """Attempt stochastic 3D cohesin moves on the GPU"""

    LEF_typeid = system.state.bond_types.index('LEF')
    backbone_typeid = system.state.bond_types.index('Backbone')
    
    with system.state.gpu_local_snapshot as local_snap:
        bond_ids = xp.asarray(local_snap.bonds.typeid)
        anchors = _get_lef_anchors(bond_ids, LEF_typeid, leg_off_rate)
        
        N = int(bond_ids.size)
        rng = xp.random.random(N).astype(xp.float32)

        groups = local_snap.bonds.group._coerce_to_ndarray()
        tags = local_snap.particles.tag._coerce_to_ndarray()
        rtags = local_snap.particles.rtag._coerce_to_ndarray()

        is_backbone = xp.equal(bond_ids, backbone_typeid)
        backbone_ids = groups[is_backbone]
        
        N_min = int(xp.amin(backbone_ids))
        N_max = int(xp.amax(backbone_ids))

        with neighbor_list.gpu_local_nlist_arrays as data:
            nlist = data.nlist._coerce_to_ndarray()
            n_neigh = data.n_neigh._coerce_to_ndarray()
            head_list = data.head_list._coerce_to_ndarray()
            
            num_blocks = (N+threads_per_block-1) // threads_per_block
            
            _single_leg_search(
                (num_blocks,),
                (threads_per_block,),
                (N, N_min, N_max, cutoff,
                 nlist, n_neigh, head_list, tags, rtags,
                 anchors, rng,
                 groups)
            )


def _get_lef_anchors(bond_ids, type_id, leg_off_rate):
    """Stochastically unbind individual cohesin legs on the GPU"""

    is_bound = xp.equal(bond_ids, type_id)
    N_bound = int(xp.count_nonzero(is_bound))
        
    rng_left = xp.random.random(N_bound)
    rng_right = xp.random.random(N_bound)
        
    unbind_left = xp.less(rng_left, leg_off_rate)
    unbind_right = xp.less(rng_right, leg_off_rate)
        
    unbind = xp.logical_and(unbind_left, unbind_right)

    unbind_left = xp.logical_and(unbind_left, xp.logical_not(unbind))
    unbind_right = xp.logical_and(unbind_right, xp.logical_not(unbind))
	
    anchors = xp.zeros(int(bond_ids.size), dtype=xp.int32)
	
    anchors[is_bound] = xp.where(unbind_right, 1, anchors[is_bound])
    anchors[is_bound] = xp.where(unbind_left, -1, anchors[is_bound])
    
    return anchors
