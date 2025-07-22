extern "C" {
__global__ void _harmonic_distance_filter(
        const unsigned int N,
        const double k_stretch,
        const double rest_length,
        const double* rng,
        const double* hdims,
        const double* positions,
        const unsigned int* rtags,
        const int* old_bonds,
        int* new_bonds) {

    unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= N)
        return;

    int2* old_bond = (int2*) old_bonds;
    int2* new_bond = (int2*) new_bonds;

    if ( (old_bond[i].x == -1) || (old_bond[i].y == -1) )
        return;

    if ( (new_bond[i].x == -1) || (new_bond[i].y == -1) )
        return;
    
    double3* hdim = (double3*) hdims;
    double3* position = (double3*) positions;
    
    bool has_moved_left = (old_bond[i].x != new_bond[i].x);
    bool has_moved_right = (old_bond[i].y != new_bond[i].y);
    
    auto harmonic_work = [=](unsigned int& i, unsigned int& j) {
    
        double dx = position[j].x - position[i].x;
        double dy = position[j].y - position[i].y;
        double dz = position[j].z - position[i].z;
		
        dx -= (copysign(hdim->x, dx-hdim->x) + copysign(hdim->x, dx+hdim->x));
        dy -= (copysign(hdim->y, dy-hdim->y) + copysign(hdim->y, dy+hdim->y));
        dz -= (copysign(hdim->z, dz-hdim->z) + copysign(hdim->z, dz+hdim->z));
        
        double step_size = sqrt(dx*dx + dy*dy + dz*dz);
        double harmonic_force = k_stretch * (step_size - rest_length);
        
        return harmonic_force * step_size;
    };

    if ( has_moved_left ) {
        unsigned int rtag1 = rtags[old_bond[i].x];
        unsigned int rtag2 = rtags[new_bond[i].x];
        
        if ( rng[i] > exp(-harmonic_work(rtag1, rtag2)) )
            new_bond[i].x = old_bond[i].x;
    }
    
    if ( has_moved_right ) {
        unsigned int rtag1 = rtags[old_bond[i].y];
        unsigned int rtag2 = rtags[new_bond[i].y];
        
        if ( rng[i] > exp(-harmonic_work(rtag1, rtag2)) )
            new_bond[i].y = old_bond[i].y;
    }
}

__global__ void _single_leg_search(
        const unsigned int N,
        const unsigned int N_min,
        const unsigned int N_max,
        const unsigned int cutoff,
        const unsigned int* nlist,
        const unsigned int* nns,
        const unsigned long* heads,
        const unsigned int* tags,
        const unsigned int* rtags,
        const int* anchors,
        const float* rng,
        unsigned int* bonds) {

    unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= N)
        return;

    if (anchors[i] == 0)
        return;

    uint2* bond = (uint2*) bonds;

    unsigned int tag_i = (anchors[i] == 1) ? bond[i].x : bond[i].y;
    unsigned int tag_j = (anchors[i] == 1) ? bond[i].y : bond[i].x;
    
    unsigned int rtag_i = rtags[tag_i];
    unsigned int nn = nns[rtag_i];
    
    if (nn == 0)
        return;
        
    unsigned long head = heads[rtag_i];
    unsigned int id_j = (unsigned int) (rng[i] * nn);

    unsigned int new_rtag_j = nlist[head + id_j];
    unsigned int new_tag_j = tags[new_rtag_j];
    
    int delta = (int) (new_tag_j - tag_j);

    if ( (new_tag_j < N_min) || (new_tag_j >= N_max) )
        return;

    if (fabsf(delta) > cutoff)
        return;
    
    bond[i] = (anchors[i] == 1) ? make_uint2(tag_i, new_tag_j) : make_uint2(new_tag_j, tag_i);
}
}
