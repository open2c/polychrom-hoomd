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


def set_excluded_volume(nlist, mode_integ, seed=0, width=1000, **force_dict):
    """Set (soft) excluded-volume repulsion based on choice of thermostat"""

    force_params = force_dict['Non-bonded forces']['Repulsion']
    cutoff = force_params['Cutoff']
    
    if mode_integ == "Langevin":
        excluded_force = md.pair.table(width=width, nlist=nlist, name="excluded")
        
        for t1 in force_params['Matrix']:
            for t2 in force_params['Matrix'][t1]:
                epsilon = force_params['Matrix'][t1][t2]
                excluded_force.pair_coeff.set(t1, t2,
                                              func=DPD_func if force_params['Type'] == "DPD" else poly_func,
                                              rmin=0., rmax=cutoff,
                                              coeff=dict(epsilon=epsilon))
                                 
    elif mode_integ == "DPD":
        excluded_force = md.pair.dpd(nlist=nlist, kT=1.0, r_cut=cutoff, seed=seed)
        
        for t1 in force_params['Matrix']:
            for t2 in force_params['Matrix'][t1]:
                epsilon = force_params['Matrix'][t1][t2]
                excluded_force.pair_coeff.set(t1, t2, A=epsilon, gamma=1)
                
    else:
        raise NotImplementedError("Unsupported integration type: %s" % mode_integ)
                                
    return excluded_force


def set_specific_attraction(nlist, pad=1e-2, width=1000, **force_dict):
    """Set type-dependent attraction potentials"""

    force_params = force_dict['Non-bonded forces']['Attraction']
    cutoff = force_params['Cutoff']

    if force_params['Type'] == "PSW":
        specific_force = md.pair.table(width=width, nlist=nlist, name="specific")
        
        for t1 in force_params['Matrix']:
            for t2 in force_params['Matrix'][t1]:
                epsilon = force_params['Matrix'][t1][t2]
                specific_force.pair_coeff.set(t1, t2,
                                              func=CSW_func,
                                              rmin=min(1+pad, cutoff) if epsilon > 0 else pad,
                                              rmax=cutoff+pad if epsilon > 0 else 2*pad,
                                              coeff=dict(epsilon=epsilon, rcut=cutoff))
    
    else:
        raise NotImplementedError("Unsupported attraction type: %s" % force_params['Type'])
    
    return specific_force
    

def set_bonds(**force_dict):
    """Set bond potentials for both polymer backbone and LEF anchors"""

    force_params = force_dict['Bonded forces']
    
    types = [params["Type"] for params in force_params.values()]
    btype = types[0]
    
    assert (all(t == btype for t in types))

    if btype == "Harmonic":
        bond_force = md.bond.harmonic()

        for bond, params in force_params.items():
            r0 = params['Rest length']
            k_stretch = 1./params['Wiggle distance']**2

            bond_force.bond_coeff.set(bond, k=k_stretch, r0=r0)
        
    else:
        raise NotImplementedError("Unsupported bond type: %s" % force_params['Type'])
    
    return bond_force


def set_angles(width=1000, **force_dict):
    """Set backbone bending potentials"""

    force_params = force_dict['Angular forces']
    
    types = [params["Type"] for params in force_params.values()]
    atype = types[0]
    
    assert (all(t == atype for t in types))
    
    if atype == "KG":
        angle_force = md.angle.table(width=width)

        for angle, params in force_params.items():
            kappa = params['Stiffness']
            angle_force.angle_coeff.set(angle, func=kg_func, coeff=dict(kappa=kappa))
    
    else:
        raise NotImplementedError("Unsupported angle type: %s" % force_params['Type'])
        
    return angle_force


def set_confinement(R, **force_dict):
    """Set rigid container with (pseudo) hard-body repulsion"""
    
    force_params = force_dict['External forces']['Confinement']
    exclusion_params = force_dict['Non-bonded forces']['Repulsion']

    wallstructure = md.wall.group()
    cutoff = exclusion_params['Cutoff']

    if force_params['Type'] == "Spherical":
        wallstructure.add_sphere(r=R, origin=(0, 0, 0), inside=True)
    else:
        raise NotImplementedError("Unsupported confinement type: %s" % force_params['Type'])

    wall_force = md.wall.slj(wallstructure, r_cut=2**(1/6.) * cutoff)
    
    for t1 in exclusion_params['Matrix']:
        wall_force.force_coeff.set(t1, epsilon=1.0, sigma=cutoff)
    
    return wall_force
