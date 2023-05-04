import hoomd
import gsd.hoomd

import numpy as np


def compute_LEF_pos(extruder_engine, n_tot,
                    trajectory_length, dummy_steps, bin_steps,
                    LEF_lifetime, LEF_separation, LEF_stall, TAD_size,
                    **kwargs):
    """LEF dynamics computation"""

    LEF_num = n_tot // LEF_separation
    
    birth_array = np.zeros(n_tot, dtype=np.double) + 0.1
    death_array = np.zeros(n_tot, dtype=np.double) + 1./LEF_lifetime
    stall_death_array = np.zeros(n_tot, dtype=np.double) + 1./LEF_lifetime
    pause_array = np.zeros(n_tot, dtype=np.double)

    stall_list = np.arange(0, n_tot, TAD_size)
    stall_left_array = np.zeros(n_tot, dtype=np.double)
    stall_right_array = np.zeros(n_tot, dtype=np.double)
    
    for i in stall_list:
        stall_left_array[i] = LEF_stall
        stall_right_array[i] = LEF_stall
        
    LEF_tran = extruder_engine(birth_array, death_array,
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


def update_topology(system, chromosome_sizes, bond_list, thermalize=False):
    """Update topology based on LEF positions"""
    
    snap = system.state.get_snapshot()
    snap_gsd = _get_gsd_snapshot(snap)
        
    redundant_bonds = (bond_list[:,1] - bond_list[:,0] < 2)
    LEF_bonds = bond_list[~redundant_bonds]

    chromosome_ends = np.cumsum(chromosome_sizes)
    bond_ids = np.digitize(LEF_bonds, chromosome_ends)
    
    extra_bonds = (bond_ids[:,1] != bond_ids[:,0])
    LEF_bonds = LEF_bonds[~extra_bonds]
    
    n_LEF = LEF_bonds.shape[0]
    n_non_LEF = np.count_nonzero(snap.bonds.typeid != 1)
        
    groups = np.zeros((n_non_LEF+n_LEF, 2), dtype=np.int32)
    typeids = np.zeros(n_non_LEF+n_LEF, dtype=np.int32)
    
    groups[:n_non_LEF] = snap.bonds.group[:n_non_LEF]
    groups[n_non_LEF:] = LEF_bonds
    
    typeids[:n_non_LEF] = snap.bonds.typeid[:n_non_LEF]
    typeids[n_non_LEF:] = 1

    snap_gsd.bonds.N = groups.shape[0]

    snap_gsd.bonds.group = groups
    snap_gsd.bonds.typeid = typeids

    snap = hoomd.Snapshot.from_gsd_snapshot(snap_gsd, snap.communicator)
    
    system.state.set_snapshot(snap)
    
    if thermalize:
        system.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)


def _get_gsd_snapshot(snap_hoomd):
    """Convert HOOMD snapshots to assignable GSD snapshots"""

    snap_gsd = gsd.hoomd.Frame()
    snap_gsd.configuration.box = snap_hoomd.configuration.box

    for attr in snap_gsd.__dict__:
        data_gsd = getattr(snap_gsd, attr)
        
        if hasattr(snap_hoomd, attr):
            data_hoomd = getattr(snap_hoomd, attr)

            if hasattr(data_gsd, '__dict__'):
                for prop in data_gsd.__dict__:
                    if hasattr(data_hoomd, prop):
                        setattr(data_gsd, prop, getattr(data_hoomd, prop))
        
    return snap_gsd
