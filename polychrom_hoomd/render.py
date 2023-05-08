import numpy as np
import matplotlib.pyplot as plt

from polykit.renderers import backends
from polychrom_hoomd.utils import get_chrom_bounds

from matplotlib.cm import get_cmap
from matplotlib.ticker import AutoLocator
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, ListedColormap, NoNorm


def domain_viewer(snap,
                  cmap='coolwarm',
                  numticks=10,
                  height=0.5,
                  width=10):
    """
    Visualize chromatin domains per chromosome in 1D
    """
        
    chrom_bounds = get_chrom_bounds(snap)
    chrom_lengths = np.diff(chrom_bounds, axis=1).flatten() + 1.
    
    n_chrom = chrom_bounds.shape[0]
    
    typeids = snap.particles.typeid.copy()
    vmin, vmax = typeids.min(), typeids.max()

    fig = plt.figure(figsize=(width, height*n_chrom))
    
    heights = np.linspace(1, 0, num=n_chrom)
    lengths = chrom_lengths/chrom_lengths.max()

    for i in range(n_chrom):
        map_name = "chromosome %d" % (i+1)
        ax = fig.add_axes([(1-lengths[i])/2., heights[i], lengths[i], height/n_chrom])
        
        ax.set_title(map_name)
        
        types = typeids[chrom_bounds[i,0]:chrom_bounds[i,1]+1]
        
        colors = np.ones((types.shape[0], 4))
        colors[:,:3] = get_cmap(cmap)(Normalize(vmin=vmin, vmax=vmax)(types))[:,:3]

        map = ListedColormap(colors, name=map_name)
        
        try:
            plt.register_cmap(name=map_name, cmap=map)
            
        except ValueError:
            pass
            
        plt.set_cmap(map)
        cb = ColorbarBase(ax, orientation='horizontal', norm=NoNorm())
    
        cb.locator = AutoLocator()
        cb.update_ticks()
    
    plt.show()


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
    
    radii = snap.particles.diameter.copy() * 0.5
    colorscale = np.zeros(snap.particles.N)
    
    bond_mask = np.ones(snap.particles.N, dtype=bool)
    polymer_mask = np.ones(snap.particles.N, dtype=bool)

    polymer_mask[bonds] = False
    bond_mask[bonds[snap.bonds.typeid>0]] = False
    
    num_unbound_atoms = np.count_nonzero(polymer_mask)

    if num_unbound_atoms > 0:
        bond_mask[-num_unbound_atoms:] = False

    if show_chromosomes:
        chrom_bounds = get_chrom_bounds(snap)
                
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
    
    radii[bond_mask] *= rescale_backbone_bonds
    colors = get_cmap(cmap)(Normalize()(colorscale))[:,:3]

    return backends.fresnel(positions, bonds, colors, radii, **kwargs)
