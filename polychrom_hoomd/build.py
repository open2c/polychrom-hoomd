import hoomd
import gsd.hoomd

import numpy as np

from . import log
from scipy.spatial import ConvexHull


def get_hoomd_device(notice_level=3):
    """
    Initialise HOOMD on the CPU or GPU, based on availability
    """
    
    try:
        device = hoomd.device.GPU(notice_level=notice_level)
        
        print("HOOMD is running on the following GPU(s):")
        print("\n".join(device.devices))
        
    except RuntimeError:
        device = hoomd.device.CPU(notice_level=notice_level)
        
        print("HOOMD is running on the CPU")
        
    return device


def get_simulation_box(box_length, pad=0):
    """Setup simulation box and initial chromatin state"""
    
    snap = gsd.hoomd.Frame()
    
    box = [box_length*(1+pad)]*3 + [0]*3
    snap.configuration.box = np.asarray(box, dtype=np.float32)
    
    return snap
    

def set_chromosomes(snap, monomer_positions, chromosome_sizes,
                    monomer_type_list=['A', 'B'],
                    bond_type_list=['Backbone'],
                    angle_type_list=['Curvature'],
                    center=True):
    """
    Set chromosome conformations and topology
    """
    
    if center:
        monomer_positions = np.asarray(monomer_positions, dtype=np.float32)
        monomer_positions -= monomer_positions.mean(axis=0, keepdims=True)
	
    update_snapshot_data(snap.particles, monomer_positions, monomer_type_list)
    set_backbone_topology(snap, chromosome_sizes, bond_type_list, angle_type_list)


def set_membrane_vertices(snap, vertex_positions,
                          vertex_type_list=['Vertices'],
                          bond_type_list=['Membrane'],
                          dihedral_type_list=['Curvature'],
                          center=True):
    """
    Set chromosome conformations and topology
    """

    if center:
        vertex_positions = np.asarray(vertex_positions, dtype=np.float32)
        vertex_positions -= vertex_positions.mean(axis=0, keepdims=True)
        
    update_snapshot_data(snap.particles, vertex_positions, vertex_type_list)
    set_membrane_topology(snap, vertex_type_list, bond_type_list, dihedral_type_list)

    
def set_backbone_topology(snap, chromosome_sizes, bond_type_list, angle_type_list):
    """
    Set backbone bonds/angles
    """

    chromosome_ends = np.cumsum(chromosome_sizes)
    monomer_ids = np.arange(chromosome_ends[-1])

    backbone_bonds = list(zip(monomer_ids[:-1], monomer_ids[1:]))
    backbone_angles = list(zip(monomer_ids[:-2], monomer_ids[1:-1], monomer_ids[2:]))
        
    snap.bonds.types = bond_type_list
    snap.angles.types = angle_type_list

    for end in chromosome_ends[::-1][1:]:
        backbone_bonds.pop(end-1)
        backbone_angles.pop(end-1)
        backbone_angles.pop(end-2)

    update_snapshot_data(snap.bonds, backbone_bonds, bond_type_list)
    update_snapshot_data(snap.angles, backbone_angles, angle_type_list)
	

def set_membrane_topology(snap, vertex_type_list, bond_type_list, dihedral_type_list):
    """
    Set membrane bonds/dihedrals
    """

    vertex_typeid = snap.particles.types.index(vertex_type_list[0])
    vertex_ids = np.flatnonzero(snap.particles.typeid == vertex_typeid)
    
    vertex_positions = snap.particles.position[vertex_ids]
    hull = ConvexHull(vertex_positions)

    vertex_offset = vertex_ids.min()
    n_simplices = len(hull.simplices)
    
    membrane_bonds = []
    membrane_dihedrals = []

    for i in range(n_simplices):
        nb = hull.neighbors[i]
        simp = hull.simplices[i] + vertex_offset

        for j in range(3):
            if nb[j] > i:
                edge = np.delete(simp, j)
                simp_nb = hull.simplices[nb[j]] + vertex_offset

                k = set(simp_nb) - set(edge)
                dihed = [simp[j]] + list(edge) + list(k)

                membrane_bonds.append(edge)
                membrane_dihedrals.append(dihed)
				
    update_snapshot_data(snap.bonds, membrane_bonds, bond_type_list)
    update_snapshot_data(snap.dihedrals, membrane_dihedrals, dihedral_type_list)


def update_snapshot_data(snapshot_data, new_data, new_type_list):
    """
    Append new particles/bonds/angles/dihedrals to snapshot
    """

    number_of_entries = len(new_data)
	
    typeids = snapshot_data.typeid if snapshot_data.N else []
    types = snapshot_data.types if snapshot_data.N else []

    typeids = list(typeids) + [len(types)]*number_of_entries
    types = list(types) + new_type_list

    snapshot_data.types = types
    snapshot_data.typeid = np.asarray(typeids, dtype=np.uint32)
    
    if hasattr(snapshot_data, 'position'):
        positions = snapshot_data.position if snapshot_data.N else []
        positions = list(positions) + list(new_data)
		
        snapshot_data.position = np.asarray(positions, dtype=np.float32)
        snapshot_data.diameter = np.ones(len(positions), dtype=np.float32)
	
    elif hasattr(snapshot_data, 'group'):
        groups = snapshot_data.group if snapshot_data.N else []
        groups = list(groups) + list(new_data)

        snapshot_data.group = np.asarray(groups, dtype=np.uint32)
        
    snapshot_data.N += number_of_entries
    
    
def get_thomson_distribution(N, radius=1, steps=1e5):
    """
    Generate uniform spherical vertex distribution through Thomson relaxation
    """
    
    hoomd_device = get_hoomd_device()

    vertex_positions = np.random.randn(N, 3)
    vertex_positions /= np.linalg.norm(vertex_positions, axis=1, keepdims=True) / radius
    
    snapshot = get_simulation_box(box_length=5*radius)
    update_snapshot_data(snapshot.particles, vertex_positions, ['Vertices'])

    system = hoomd.Simulation(device=hoomd_device)
    system.create_state_from_snapshot(snapshot)
    
    sphere = hoomd.md.manifold.Sphere(r=radius)
    nve = hoomd.md.methods.rattle.NVE(filter=hoomd.filter.All(), manifold_constraint=sphere)
    
    nl = hoomd.md.nlist.Cell(buffer=0.4)
    
    coulomb_force = hoomd.md.pair.Yukawa(default_r_cut=2*radius, nlist=nl)
    coulomb_force.params[('Vertices', 'Vertices')] = dict(epsilon=1.0, kappa=0.)

    fire = hoomd.md.minimize.FIRE(dt=2e-5, methods=[nve], forces=[coulomb_force],
                                  force_tol=5e-2, angmom_tol=5e-2, energy_tol=5e-2)
    logger = log.get_logger(system, quantities=['potential_energy'])
    
    system.operations.integrator = fire
    system.operations.writers.append(log.table_formatter(logger, period=5e3))
    
    system.run(steps)
    
    relaxed_snapshot = system.state.get_snapshot()

    return relaxed_snapshot.particles.position
