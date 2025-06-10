extern "C" {
__global__ void _harmonic_boltzmann_filter(
        const unsigned int N,
        const double sigma2,
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

    if ( has_moved_left || has_moved_right ) {
        double delta_r2 = 0.;
        
        for ( unsigned int j = 0; j < 2; ++j ) {
            int2 tags = ((j==0) ? old_bond[i] : new_bond[i]);
    
            unsigned int rtag1 = rtags[tags.x];
            unsigned int rtag2 = rtags[tags.y];
			
            double dx = position[rtag2].x - position[rtag1].x;
            double dy = position[rtag2].y - position[rtag1].y;
            double dz = position[rtag2].z - position[rtag1].z;
			
            dx -= (copysign(hdim->x, dx-hdim->x) + copysign(hdim->x, dx+hdim->x));
            dy -= (copysign(hdim->y, dy-hdim->y) + copysign(hdim->y, dy+hdim->y));
            dz -= (copysign(hdim->z, dz-hdim->z) + copysign(hdim->z, dz+hdim->z));

            double r2 = dx*dx + dy*dy + dz*dz;
            delta_r2 += ((j==0) ? r2 : -r2);
        }
        
        double boltzmann_weight = exp(delta_r2/sigma2);

        if ( rng[i] > boltzmann_weight ) {
            if ( has_moved_left )
                new_bond[i].x = old_bond[i].x;
            if ( has_moved_right )
                new_bond[i].y = old_bond[i].y;
        }
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
    
    bool is_left_anchor = (anchors[i] == 1);

    unsigned int tag_i = is_left_anchor ? bond[i].x : bond[i].y;
    unsigned int tag_j = is_left_anchor ? bond[i].y : bond[i].x;
    
    unsigned int rtag_i = rtags[tag_i];
    unsigned int nn = nns[rtag_i];
    
    if (nn == 0)
        return;
        
    unsigned long head = heads[rtag_i];
    unsigned int id_j = (unsigned int) (rng[i] * nn);

    unsigned int new_rtag_j = nlist[head + id_j];
    unsigned int new_tag_j = tags[new_rtag_j];
    
    int delta = (int) (new_tag_j - tag_j);

    if ((new_tag_j < N_min) || (new_tag_j >= N_max))
        return;
    if (fabsf(delta) > cutoff)
        return;
    
    bond[i] = is_left_anchor ? make_uint2(tag_i, new_tag_j) : make_uint2(new_tag_j, tag_i);
}
}
