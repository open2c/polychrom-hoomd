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
                  w=10,
                  h=0.25):
    """
    Visualize chromatin domains in 1D
    """
    
    plt.figure(figsize=(w, h))
    
    colors = np.ones((snap.particles.typeid.shape[0], 4))
    colors[:,:3] = get_cmap(cmap)(Normalize()(snap.particles.typeid))[:,:3]

    map = ListedColormap(colors, name="map")
    
    plt.register_cmap(name="map", cmap=map)
    plt.set_cmap(map)
    
    cb = ColorbarBase(plt.gca(), orientation='horizontal', norm=NoNorm())
    
    cb.locator = AutoLocator()
    cb.update_ticks()
    
    plt.show()


def fresnel(snap,
            cmap='viridis',
            show_compartments=False,
            **kwargs):
    """
    Wrapper around polykit.renderers.backends for HooMD rendering using the Fresnel library
    """

    bonds = snap.bonds.group.copy()
    positions = snap.particles.position.copy()

    if show_compartments:
        colors = get_cmap(cmap)(Normalize()(snap.particles.typeid))[:,:3]
    else:
        colors = get_cmap(cmap)(Normalize()(np.arange(snap.particles.N)))[:,:3]
        
    return backends.fresnel(positions, bonds, colors, **kwargs)
