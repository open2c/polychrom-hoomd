import numpy as np
import matplotlib.pyplot as plt

from polykit.renderers import backends

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
        
    chrom_bounds = _get_chrom_bounds(snap)
    chrom_lengths = np.diff(chrom_bounds, axis=1).flatten()
    
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
    diameters = snap.particles.diameter[bonds].mean(axis=1)
    
    colorscale = np.zeros(snap.particles.N)

    if show_chromosomes:
        chrom_bounds = _get_chrom_bounds(snap)
                
        for i, bounds in enumerate(chrom_bounds):
            colorscale[bounds[0]:bounds[1]+1] = i+1
                        
    elif show_loops:
        loop_bounds = bonds[snap.bonds.typeid == 1]
                
        for i, bounds in enumerate(loop_bounds):
            colorscale[bounds[0]:bounds[1]+1] = i+1
                    
    elif show_compartments:
        colorscale = snap.particles.typeid.copy()
            
    else:
        colorscale = np.arange(snap.particles.N)
    
    diameters[snap.bonds.typeid == 0] *= rescale_backbone_bonds
    
    colors = get_cmap(cmap)(Normalize()(colorscale))[:,:3]

    return backends.fresnel(positions, bonds, colors, diameters, **kwargs)


def _get_chrom_bounds(snap):
    """
    Infer chromosome bounds from the snapshot topology
    """

    backbone_bonds = snap.bonds.group[snap.bonds.typeid == 0]

    bond_breaks, = np.nonzero(backbone_bonds[1:,0] != backbone_bonds[:-1,1])
    chrom_list = np.split(backbone_bonds, bond_breaks+1)
    
    chrom_bounds = np.zeros((len(chrom_list), 2), dtype=np.int32)
    
    for i, bonds in enumerate(chrom_list):
        chrom_bounds[i] = bonds[0,0], bonds[-1,1]

    return chrom_bounds
