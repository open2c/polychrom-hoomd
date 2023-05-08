import hoomd
import gsd.hoomd

import numpy as np

from polychrom_hoomd.utils import get_trans_cis_ids, get_gsd_snapshot


def compute_LEF_pos(extrusion_engine, n_tot,
                    trajectory_length, dummy_steps, bin_steps,
                    LEF_lifetime, LEF_separation, LEF_stall, TAD_size,
                    **kwargs):
    """LEF dynamics computation"""

    LEF_num = n_tot // LEF_separation
    
    birth_array = np.zeros(n_tot, dtype=np.double) + 0.1
    pause_array = np.zeros(n_tot, dtype=np.double)
        
    death_array = np.zeros(n_tot, dtype=np.double) + 1./LEF_lifetime
    stall_death_array = np.zeros(n_tot, dtype=np.double) + 1./LEF_lifetime
    
    stall_list = np.arange(0, n_tot, TAD_size)
    stall_left_array = np.zeros(n_tot, dtype=np.double)
    stall_right_array = np.zeros(n_tot, dtype=np.double)
    
    for i in stall_list:
        stall_left_array[i] = LEF_stall
        stall_right_array[i] = LEF_stall
        
    LEF_tran = extrusion_engine(birth_array, death_array,
                                stall_left_array, stall_right_array,
                                pause_array, stall_death_array,
                                LEF_num)
    
    LEF_tran.steps(dummy_steps)

    LEF_pos = np.zeros((trajectory_length, LEF_num, 2), dtype=int)
    bins = np.linspace(0, trajectory_length, bin_steps, dtype=int)

    for st,end in zip(bins[:-1], bins[1:]):
        cur = []
        
        for i in range(st, end):
            LEF_tran.steps(1)
            cur.append(np.asarray(LEF_tran.getLEFs()).T)
            
        LEF_pos[st:end] = np.asarray(cur)

    return LEF_pos


def update_topology(system, bond_list, thermalize=False):
    """Update topology based on LEF positions"""
    
    snap = system.state.get_snapshot()
    snap_gsd = get_gsd_snapshot(snap)
    
    # Discard contiguous loops
    redundant_bonds = (bond_list[:,1] - bond_list[:,0] < 2)
    LEF_bonds = bond_list[~redundant_bonds]

    # Discard trans-chromosomal loops
    bond_trans_ids, _ = get_trans_cis_ids(LEF_bonds, snap)
    
    trans_bonds = (bond_trans_ids[:,1] != bond_trans_ids[:,0])
    LEF_bonds = LEF_bonds[~trans_bonds]
    
    # Update LEF bonds
    n_LEF = LEF_bonds.shape[0]
    n_non_LEF = np.count_nonzero(snap.bonds.typeid != 1)
        
    groups = np.zeros((n_non_LEF+n_LEF, 2), dtype=np.int32)
    typeids = np.zeros(n_non_LEF+n_LEF, dtype=np.int32)
    
    groups[:n_non_LEF] = snap.bonds.group[:n_non_LEF]
    groups[n_non_LEF:] = LEF_bonds
    
    # LEF bonds should always be assigned to typeid 1
    typeids[:n_non_LEF] = snap.bonds.typeid[:n_non_LEF]
    typeids[n_non_LEF:] = 1

    # Bond resizing in HOOMD v3 requires full array reassignment
    snap_gsd.bonds.N = groups.shape[0]

    snap_gsd.bonds.group = groups
    snap_gsd.bonds.typeid = typeids

    # Configuration step/box data requires manual setting as of gsd v2.8
    snap_gsd.configuration.step = system.timestep
    snap_gsd.configuration.box = snap.configuration.box
    
    # Load snapshot and re-thermalize, if required
    snap = hoomd.Snapshot.from_gsd_snapshot(snap_gsd, snap.communicator)
    
    system.state.set_snapshot(snap)
    
    # Re-thermalize system to limit extrusion-driven temperature drift
    if thermalize:
        system.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
