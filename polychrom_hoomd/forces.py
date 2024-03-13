import numpy as np

from hoomd import md, wall
    

def DPD_func(cutoff, epsilon, width=1000):
    """DPD soft-core repulsion"""
    
    r = np.linspace(0, cutoff, width, endpoint=False)

    U = epsilon/2. * cutoff * (1. - r/cutoff)**2
    F = epsilon * (1. - r/cutoff)
    
    return U, F
    
    
def poly_func(cutoff, epsilon, width=1000):
    """Polychrom soft-core repulsion"""

    r = np.linspace(0, cutoff, width, endpoint=False)
    r_resc = r/cutoff * np.sqrt(6/7)

    U = epsilon * (1 + r_resc**12 * (r_resc**2 - 1) * (823543./46656.))
    F = epsilon * r**11 * 84 * (cutoff**2 - r**2) / cutoff**14
    
    return U, F
    

def PSW_func(r_min, r_max, epsilon, width=1000):
    """Polychrom pseudo-square-well attraction"""

    r = np.linspace(r_min, r_max, width, endpoint=False)
    
    delta = (r_max-r_min) / 2.
    r_shift = (r-r_min-delta) / delta * np.sqrt(6/7.)
    
    U = -epsilon * (1 + r_shift**12 * (r_shift**2-1) * (823543. / 46656.))
    F = epsilon*(r_max + r_min - 2*r)**11  \
            * (168*(r_max - r_min)**2 - 120*(r_max + r_min - 2*r)**2)  \
            / (r_max - r_min)**14
    
    return U, F
    
    
def kg_func(kappa, width=1000):
    """Kremer-Grest bending energy penalty function"""

    theta = np.linspace(0, np.pi, width, endpoint=True)

    U = kappa * (1. + np.cos(theta))
    tau = kappa * np.sin(theta)

    return U, tau


def harmonic_func(kappa, width=1000):
    """Harmonic bending energy penalty function"""

    theta = np.linspace(0, np.pi, width, endpoint=True)

    U = kappa/2 * (theta-np.pi)**2
    tau = kappa * (np.pi-theta)

    return U, tau
    
    
def get_repulsion_forces(nlist, **force_dict):
    """Setup (soft) excluded-volume repulsion based on choice of thermostat"""

    force = force_dict['Non-bonded forces']['Repulsion']
    cutoff = force['Cutoff']
    
    repulsion_force = md.pair.Table(nlist=nlist, default_r_cut=cutoff)
        
    for t1 in force['Matrix']:
        for t2 in force['Matrix'][t1]:
            epsilon = force['Matrix'][t1][t2]
                
            if force['Type'] == "DPD":
                U, F = DPD_func(cutoff, epsilon)
                
            elif force['Type'] == "Polychrom":
                U, F = poly_func(cutoff, epsilon)
                
            else:
                raise NotImplementedError("Unsupported repulsion force: %s" % force['Type'])
                    
            repulsion_force.params[(t1, t2)] = dict(r_min=0, U=U, F=F)
                                 
    return [repulsion_force]
    

def get_attraction_forces(nlist, **force_dict):
    """Setup type-dependent attraction potentials"""

    force = force_dict['Non-bonded forces']['Attraction']
    cutoff = force['Cutoff']
    
    repulsion_force = force_dict['Non-bonded forces']['Repulsion']
    repulsion_cutoff = repulsion_force['Cutoff']
    
    assert cutoff > repulsion_cutoff

    attraction_force = md.pair.Table(nlist=nlist, default_r_cut=cutoff)

    for t1 in force['Matrix']:
        for t2 in force['Matrix'][t1]:
            epsilon = force['Matrix'][t1][t2]
            
            if force['Type'] == "Polychrom":
                U, F = PSW_func(r_min=repulsion_cutoff, r_max=cutoff, epsilon=epsilon)
                attraction_force.params[(t1, t2)] = dict(r_min=repulsion_cutoff, U=U, F=F)
                    
            else:
                raise NotImplementedError("Unsupported attraction force: %s" % force['Type'])
            
    return [attraction_force]


def get_dpd_forces(nlist, **force_dict):
    """Setup DPD pairwise conservative/dissipative/random forces"""

    force = force_dict['Non-bonded forces']['Repulsion']
    cutoff = force['Cutoff']
        
    for _force in force_dict['Non-bonded forces'].values():
        if cutoff < _force['Cutoff']:
            cutoff = _force['Cutoff']
            
    disable_conservative = (force['Type'] != "DPD") | (cutoff != force['Cutoff'])
    
    if disable_conservative:
        print("Setting up DPD with the conservative force contribution disabled")

    dpd_force = md.pair.DPD(nlist=nlist, kT=1.0, default_r_cut=cutoff)

    for t1 in force['Matrix']:
        for t2 in force['Matrix'][t1]:
            if disable_conservative:
                dpd_force.params[(t1, t2)] = dict(A=0, gamma=1)
                
            else:
                epsilon = force['Matrix'][t1][t2]
                dpd_force.params[(t1, t2)] = dict(A=epsilon, gamma=1)
                                            
    return [dpd_force]
    

def get_bonded_forces(**force_dict):
    """Setup bonded potentials for both polymer backbone and LEF anchors"""

    bonded_forces = []
    
    force_list = force_dict['Bonded forces']
    force_types = set(force['Type'] for force in force_list.values())
    
    for force_type in force_types:
        if force_type == "Harmonic":
            harmonic_force = md.bond.Harmonic()
            
            for bond_type, force in force_list.items():
                if force['Type'] == force_type:
                    r0 = force['Rest length']
                    k_stretch = 1./force['Wiggle distance']**2

                    harmonic_force.params[bond_type] = dict(k=k_stretch, r0=r0)
                    
                else:
                    harmonic_force.params[bond_type] = dict(k=0, r0=0)
            
            bonded_forces.append(harmonic_force)
            
        elif force_type == "FENE":
            fene_force = md.bond.FENEWCA()
            
            for bond_type, force in force_list.items():
                if force['Type'] == force_type:
                    r0 = force['Bond length']
                    k = force['Attraction strength']
                    
                    sigma = force['Repulsion width']
                    epsilon = force['Repulsion strength']

                    fene_force.params[bond_type] = dict(k=k, r0=r0, epsilon=epsilon, sigma=sigma, delta=0)
                    
                else:
                    fene_force.params[bond_type] = dict(k=0, r0=1, epsilon=0, sigma=1, delta=0)
                    
            bonded_forces.append(fene_force)
            
        else:
            raise NotImplementedError("Unsupported bonded force: %s" % force_type)
    
    return bonded_forces


def get_angular_forces(width=1000, **force_dict):
    """Setup backbone bending potentials"""

    angular_forces = []
    
    force_list = force_dict['Angular forces']
    force_types = set(force['Type'] for force in force_list.values())
    
    for force_type in force_types:
        if force_type in ["KG", "Harmonic"]:
            angular_force = md.angle.Table(width=width)
            tabulated_func = kg_func if force_type == "KG" else harmonic_func

            for angle_type, force in force_list.items():
                if force['Type'] == force_type:
                    kappa = force['Stiffness']
                    U, tau = tabulated_func(kappa, width=width)

                    angular_force.params[angle_type] = dict(U=U, tau=tau)
                    
                else:
                    angular_force.params[angle_type] = dict(U=[0], tau=[0])
                    
            angular_forces.append(angular_force)
                    
        else:
            raise NotImplementedError("Unsupported angular force: %s" % force_type)
        
    return angular_forces


def get_confinement_forces(**force_dict):
    """Setup confinement fields with (pseudo) hard-body repulsion"""
    
    walls = []
    
    force_list = force_dict['External forces']['Confinement']
    repulsion_force = force_dict['Non-bonded forces']['Repulsion']
    
    cutoff = repulsion_force['Cutoff']

    for confinement_type, force in force_list.items():
        if confinement_type == "Spherical":
            walls.append(wall.Sphere(radius=force['R']))

        else:
            raise NotImplementedError("Unsupported confinement type: %s" % confinement_type)

    wall_force = md.external.wall.ForceShiftedLJ(walls=walls)
        
    for t in repulsion_force['Matrix']:
        wall_force.params[t] = dict(epsilon=1.0, sigma=cutoff, r_cut=2**(1/6.) * cutoff)
    
    return [wall_force]
