extern "C"
__global__ void _single_leg_search(
        const unsigned int N,
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
    bool is_left_anchor = (anchors[i] == -1);

    unsigned int tag_i = is_left_anchor ? bond[i].x : bond[i].y;
    unsigned int rtag_i = rtags[tag_i];
    
    unsigned int nn = nns[rtag_i];
    unsigned long head = heads[rtag_i];

    if (nn > 0) {
        unsigned int id_j = (unsigned int) (rng[i] * nn);
        
        unsigned int rtag_j = nlist[head + id_j];
        unsigned int tag_j = tags[rtag_j];

        bond[i] = is_left_anchor ? make_uint2(tag_i, tag_j) : make_uint2(tag_j, tag_i);
    }
}
