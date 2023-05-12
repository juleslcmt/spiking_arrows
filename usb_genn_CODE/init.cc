#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseSparseInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    scalar* g;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int colStride;
    scalar constantg;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, unsigned int numNeurons) {
    MergedNeuronInitGroup1 group = {spkCnt, spk, V, RefracTime, inSynInSyn0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1)));
}
__device__ __constant__ MergedNeuronInitGroup2 d_mergedNeuronInitGroup2[5];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup2 group = {spkCnt, spk, V, RefracTime, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup2, &group, sizeof(MergedNeuronInitGroup2), idx * sizeof(MergedNeuronInitGroup2)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup0 d_mergedSynapseConnectivityInitGroup0[3];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup0 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup0, &group, sizeof(MergedSynapseConnectivityInitGroup0), idx * sizeof(MergedSynapseConnectivityInitGroup0)));
}
__device__ __constant__ MergedSynapseSparseInitGroup0 d_mergedSynapseSparseInitGroup0[3];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int colStride, scalar constantg) {
    MergedSynapseSparseInitGroup0 group = {rowLength, ind, g, rowStride, numSrcNeurons, numTrgNeurons, colStride, constantg, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseSparseInitGroup0, &group, sizeof(MergedSynapseSparseInitGroup0), idx * sizeof(MergedSynapseSparseInitGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {307200, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {614400, 614496, 921696, 921792, 921888, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID0[] = {921984, 1229184, 1536384, };
__device__ unsigned int d_mergedSynapseSparseInitGroupStartID0[] = {0, 96, 192, };

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 96 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 307200) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            // current source variables
        }
    }
    // merged1
    if(id >= 307200 && id < 614400) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 307200;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                scalar initVal;
                initVal = (-6.50000000000000000e+01f);
                group->V[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->RefracTime[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
            }
            // current source variables
        }
    }
    // merged2
    if(id >= 614400 && id < 921984) {
        unsigned int lo = 0;
        unsigned int hi = 5;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedNeuronInitGroupStartID2[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[lo - 1]; 
        const unsigned int groupStartID = d_mergedNeuronInitGroupStartID2[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                scalar initVal;
                initVal = (-6.50000000000000000e+01f);
                group->V[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->RefracTime[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
            }
             {
                group->inSynInSyn1[lid] = 0.000000000e+00f;
            }
            // current source variables
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // merged0
    if(id >= 921984 && id < 1843584) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseConnectivityInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseConnectivityInitGroup0 *group = &d_mergedSynapseConnectivityInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseConnectivityInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            // Build sparse connectivity
            while(true) {
                do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = lid;
                    group->rowLength[lid]++;
                }
                while(false);
                break;
                
            }
        }
    }
    
}
extern "C" __global__ void initializeSparseKernel() {
    const unsigned int id = 96 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shRowLength[96];
    // merged0
    if(id < 288) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseSparseInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseSparseInitGroup0 *group = &d_mergedSynapseSparseInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseSparseInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        const unsigned int numBlocks = (group->numSrcNeurons + 96 - 1) / 96;
        unsigned int idx = lid;
        for(unsigned int r = 0; r < numBlocks; r++) {
            const unsigned numRowsInBlock = (r == (numBlocks - 1)) ? ((group->numSrcNeurons - 1) % 96) + 1 : 96;
            __syncthreads();
            if (threadIdx.x < numRowsInBlock) {
                shRowLength[threadIdx.x] = group->rowLength[(r * 96) + threadIdx.x];
            }
            __syncthreads();
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(lid < shRowLength[i]) {
                     {
                        scalar initVal;
                        initVal = group->constantg;
                        group->g[(((r * 96) + i) * group->rowStride) + lid] = initVal;
                    }
                }
                idx += group->rowStride;
            }
        }
    }
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
     {
        const dim3 threads(96, 1);
        const dim3 grid(19204, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
     {
        const dim3 threads(96, 1);
        const dim3 grid(3, 1);
        initializeSparseKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
