import numpy as np
import polychrom_hoomd.utils as utils

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from polykit.renderers import backends, viewers


def domain_viewer(snap, cmap='coolwarm', **kwargs):
    """
    Visualize chromatin domains per chromosome in 1D
    """
        
    chrom_bounds = utils.get_chrom_bounds(snap)
    chrom_lengths = np.diff(chrom_bounds, axis=1).flatten() + 1
        
    typeids = snap.particles.typeid.copy()
    vmin, vmax = typeids.min(), typeids.max()
    
    num_monomers = sum(chrom_lengths)
    
    colors = get_cmap(cmap)(Normalize(vmin=vmin, vmax=vmax)(typeids))[:num_monomers]
    colors[:, 3] = 1.
    
    viewers.chromosome_viewer(chrom_lengths, colors, **kwargs)


def fresnel(snap,
            cmap='viridis',
            rescale_backbone_bonds=1.,
            show_chromosomes=False,
            show_compartments=False,
            show_loops=False,
            **kwargs):
    """
    Wrapper around polykit.renderers.backends for HooMD rendering using the Fresnel library
    """

    bonds = snap.bonds.group.copy()
    positions = snap.particles.position.copy()
        
    bond_mask = np.ones(snap.particles.N, dtype=bool)
    polymer_mask = np.ones(snap.particles.N, dtype=bool)

    polymer_mask[bonds] = False
    bond_mask[bonds[snap.bonds.typeid > 0]] = False
    
    bond_mask[polymer_mask] = False
    positions[~polymer_mask] = utils.unwrap_coordinates(snap, ~polymer_mask)
    
    radii = snap.particles.diameter.copy() * 0.5
    radii[bond_mask] *= rescale_backbone_bonds

    colorscale = np.zeros(snap.particles.N)

    if show_chromosomes:
        chrom_bounds = utils.get_chrom_bounds(snap)
                
        for i, bounds in enumerate(chrom_bounds):
            colorscale[bounds[0]:bounds[1]+1] = i+1
                        
    elif show_loops:
        loop_bounds = bonds[snap.bonds.typeid==1]
                
        for i, bounds in enumerate(loop_bounds):
            colorscale[bounds[0]:bounds[1]+1] = i+1
                    
    elif show_compartments:
        colorscale = snap.particles.typeid.copy()
            
    else:
        colorscale = np.arange(snap.particles.N)
    
    colors = get_cmap(cmap)(Normalize()(colorscale))

    return backends.Fresnel(positions, bonds, colors, radii, **kwargs)
