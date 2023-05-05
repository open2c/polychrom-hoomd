import hoomd
import gsd.hoomd

import numpy as np
    

def get_hoomd_device(notice_level=3):
    """Initialise HOOMD on the CPU or GPU, based on availability"""
    
    try:
        device = hoomd.device.GPU(notice_level=notice_level)
        print("HOOMD is running on the following GPU(s):\n"
              "\n".join(device.get_available_devices()))
        
    except RuntimeError:
        device = hoomd.device.CPU(notice_level=notice_level)
        print("HOOMD is running on the CPU")
        
    return device


def get_simulation_box(box_length, pad=0):
    """Setup simulation box and initial chromatin state"""
    
    snap = gsd.hoomd.Frame()
    
    snap.configuration.dimensions = 3
    snap.configuration.box = [box_length*(1+pad)]*3 + [0]*3
    
    return snap
    

def set_chains(snap, monomer_positions, chromosome_sizes,
               monomer_type_list=['A', 'B'],
               bond_type_list=['Backbone'],
               angle_type_list=['Curvature'],
               center=True):
    """Set chromosome conformations and topology"""
    
    number_of_monomers = monomer_positions.shape[0]
    monomer_positions = monomer_positions.astype(np.float32)
    
    if center:
        monomer_positions -= monomer_positions.mean(axis=0, keepdims=True)
                        
    snap.particles.N = number_of_monomers
    snap.particles.types = monomer_type_list
        
    snap.particles.position = monomer_positions
    
    snap.particles.typeid = np.zeros(number_of_monomers, dtype=np.int32)
    snap.particles.diameter = np.ones(number_of_monomers, dtype=np.float32)
    
    set_backbone_topology(snap, chromosome_sizes, bond_type_list, angle_type_list)
        
    
def set_backbone_topology(snap, chromosome_sizes, bond_type_list, angle_type_list):
    """Set backbone bonds/angles"""

    chromosome_ends = np.cumsum(chromosome_sizes)
    monomer_ids = np.arange(chromosome_ends[-1])

    bonds = list(zip(monomer_ids[:-1], monomer_ids[1:]))
    angles = list(zip(monomer_ids[:-2], monomer_ids[1:-1], monomer_ids[2:]))
        
    snap.bonds.types = bond_type_list
    snap.angles.types = angle_type_list

    for end in chromosome_ends[::-1][1:]:
        bonds.pop(end-1)
        
        angles.pop(end-1)
        angles.pop(end-2)

    number_of_bonds = len(bonds)
    number_of_angles = len(angles)

    snap.bonds.N = number_of_bonds
    snap.angles.N = number_of_angles

    snap.bonds.group = np.asarray(bonds, dtype=np.int32)
    snap.bonds.typeid = np.zeros(number_of_bonds, dtype=np.int32)

    snap.angles.group = np.asarray(angles, dtype=np.int32)
    snap.angles.typeid = np.zeros(number_of_angles, dtype=np.int32)


def set_binders(snap, binder_positions,
                binder_type_list=['Binders']):
    """Set initial binder configuration"""
    
    number_of_binders = binder_positions.shape[0]
    binder_positions = binder_positions.astype(np.float32)
    
    if snap.particles.N == 0:
        raise RuntimeError("Cannot assign binders before chromosomes")
    
    number_of_particles = number_of_binders + snap.particles.N
            
    positions = np.zeros((number_of_particles, 3), dtype=np.float32)
    typeids = np.zeros(number_of_particles, dtype=np.int32)
    
    positions[:snap.particles.N] = snap.particles.position
    positions[snap.particles.N:] = binder_positions

    typeids[:snap.particles.N] = snap.particles.typeid
    typeids[snap.particles.N:] = len(snap.particles.types)

    snap.particles.N = number_of_particles
    snap.particles.types.extend(binder_type_list)
        
    snap.particles.position = positions
    snap.particles.typeid = typeids

    snap.particles.diameter = np.ones(number_of_particles, dtype=np.float32)
