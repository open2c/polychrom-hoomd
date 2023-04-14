import numpy as np
from hoomd import md


def kg_func(theta, kappa):
    """Kremer-Grest bending energy penalty function"""

    V = kappa * (1. + np.cos(theta))
    T = kappa * np.sin(theta)

    return V, T
    
    
def poly_func(r, rmin, rmax, epsilon):
    """Polychrom soft-core repulsion"""

    term = (r* np.sqrt(6/7)) / rmax
    
    V = epsilon * (1 + (term**12) * (term**2 - 1) * (823543./46656.))
    F = -epsilon * (12.0*r**13/rmax**14 + 84.0*r**11*(0.857142857142857*r**2/rmax**2 - 1)/rmax**12)
    
    return V, F


def DPD_func(r, rmin, rmax, epsilon):
    """DPD soft-core repulsion"""
    
    V = epsilon * rmax * (1. - r/rmax)**2
    F = epsilon * (1. - r/rmax)
    
    return V, F
    
    
def CSW_func(r, rmin, rmax, epsilon, rcut, n=2500, m=2500):
    """Continuous square-well potential (see https://doi.org/10.1080/00268976.2018.1481232)"""
    
    term = np.exp(-m*(r-rmin)*(r-rcut))
    
    V = epsilon/2. * ((rmin/r)**n + (1.-term)/(1.+term) - 1)
    F = epsilon * (n/2.*(rmin/r)**(n+1) - m*(2*r-rcut-rmin)*term/(1.+term)**2)
    
    return V, F


def set_excluded_volume(nlist, mode_integ,
                        seed=0,
                        epsilon_exc=10,
                        mode_exc="dpd",
                        **kwargs):
    """Set (soft) excluded-volume repulsion based on choice of thermostat"""

    if mode_integ == "langevin":
        poly_exc = md.pair.table(width=1000, nlist=nlist, name="exc")
        poly_exc.pair_coeff.set(['A', 'B'], ['A', 'B'],
                                 func=DPD_func if mode_exc == "dpd" else poly_func,
                                 rmin=0., rmax=1., coeff=dict(epsilon=epsilon_exc))
                                 
    elif mode_integ == "dpd":
        poly_exc = md.pair.dpd(nlist=nlist, kT=1.0, r_cut=1.0, seed=seed)
        poly_exc.pair_coeff.set(['A', 'B'], ['A', 'B'],
                                A=epsilon_exc, gamma=1)
                                
    return poly_exc


def set_PSW_attraction(nlist,
                       epsilon_att=0.20, r_att=1.5, pad=1e-2,
                       **kwargs):
    """Set pseudo-square-well attraction potential"""

    # Set chromosome specific attraction
    poly_att = md.pair.table(width=1000, nlist=nlist)
    
    poly_att.pair_coeff.set(['A'], ['A', 'B'],
                            func=CSW_func, rmin=pad, rmax=2*pad,
                            coeff=dict(epsilon=0, rcut=1))
    poly_att.pair_coeff.set(['B'], ['B'],
                            func=CSW_func, rmin=min(1+pad, r_att), rmax=r_att+pad,
                            coeff=dict(epsilon=epsilon_att, rcut=r_att))

    return poly_att
    

def set_poly_bonds(wiggle_dist=0.1, wiggle_dist_smc=0.2, r0_smc=0.5,
                   **kwargs):
    """Set harmonic bonds for both polymer backbone and LEF anchors"""

    k_stretch = 1./wiggle_dist**2
    k_stretch_smc = 1./wiggle_dist_smc**2

    bond_harm = md.bond.harmonic()
        
    bond_harm.bond_coeff.set('poly_bonds', k=k_stretch, r0=1.)
    bond_harm.bond_coeff.set('SMC_bonds', k=k_stretch_smc, r0=r0_smc)
    
    return bond_harm


def set_poly_angles(k_bend=1.5, **kwargs):
    """Set polymer bending potentials"""

    angle_kg = md.angle.table(width=1000)
    angle_kg.angle_coeff.set('poly_angles', func=kg_func, coeff=dict(kappa=k_bend))
    
    return angle_kg


def set_SLJ_sphere(R):
    """Set spherical container with (pseudo) hard-body repulsion"""

    wallstructure = md.wall.group()
    wallstructure.add_sphere(r=R, origin=(0, 0, 0), inside=True)

    wall_force = md.wall.slj(wallstructure, r_cut=2**(1/6.))
    wall_force.force_coeff.set(['A', 'B'], epsilon=1.0, sigma=1.0)
    
    return wall_force
