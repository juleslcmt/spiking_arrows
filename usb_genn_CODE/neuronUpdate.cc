#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    uint32_t* input;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    float* inSynInSyn1;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    scalar TauM;
    scalar Vthresh;
    scalar ExpTC;
    scalar Rmembrane;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup1
 {
    unsigned int* spkCnt;
    
}
;
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[3];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0)));
}
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup1 d_mergedNeuronSpikeQueueUpdateGroup1[4];
void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup1 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup1, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup1), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, uint32_t* input, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {spkCnt, spk, input, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {spkCnt, spk, V, RefracTime, inSynInSyn0, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup2 d_mergedNeuronUpdateGroup2[5];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, float* inSynInSyn1, uint32_t* recordSpk, unsigned int numNeurons, scalar TauM, scalar Vthresh, scalar ExpTC, scalar Rmembrane) {
    MergedNeuronUpdateGroup2 group = {spkCnt, spk, V, RefracTime, inSynInSyn0, inSynInSyn1, recordSpk, numNeurons, TauM, Vthresh, ExpTC, Rmembrane, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &group, sizeof(MergedNeuronUpdateGroup2), idx * sizeof(MergedNeuronUpdateGroup2)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0inputToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, input)));
}

void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, recordSpk)));
}

void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, recordSpk)));
}

void pushMergedNeuronUpdate2recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup2) * (idx)) + offsetof(MergedNeuronUpdateGroup2, recordSpk)));
}

__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {307200, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID2[] = {614400, 614496, 921696, 921792, 921888, };

extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id < 3) {
        struct MergedNeuronSpikeQueueUpdateGroup0 *group = &d_mergedNeuronSpikeQueueUpdateGroup0[id - 0]; 
        group->spkCnt[0] = 0;
    }
    if(id >= 3 && id < 7) {
        struct MergedNeuronSpikeQueueUpdateGroup1 *group = &d_mergedNeuronSpikeQueueUpdateGroup1[id - 3]; 
        group->spkCnt[0] = 0;
    }
}

extern "C" __global__ void updateNeuronsKernel(float t, unsigned int recordingTimestep)
 {
    const unsigned int id = 96 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[96];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0) {
        shSpkCount = 0;
    }
    
    __shared__ uint32_t shSpkRecord[3];
    if (threadIdx.x < 3) {
        shSpkRecord[threadIdx.x] = 0;
    }
    __syncthreads();
    // merged0
    if(id < 307200) {
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        
        if(lid < group->numNeurons) {
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (
            group->input[lid / 32] & (1 << (lid % 32))
            ) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord[threadIdx.x / 32], 1 << (threadIdx.x % 32));
            }
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 3) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord[threadIdx.x];
            }
        }
    }
    // merged1
    if(id >= 307200 && id < 614400) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 307200;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar lRefracTime = group->RefracTime[lid];
            
            float Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn0[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if (lRefracTime <= 0.0f) {
              scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (1.00000000000000006e-01f)) + (-6.50000000000000000e+01f);
              lV = alpha - ((3.67879441171442334e-01f) * (alpha - lV));
            }
            else {
              lRefracTime -= DT;
            }
            
            // test for and register a true spike
            if (lRefracTime <= 0.0f && lV >= (-5.95000000000000000e+01f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord[threadIdx.x / 32], 1 << (threadIdx.x % 32));
                // spike reset code
                lV = (-6.50000000000000000e+01f);
                lRefracTime = (0.00000000000000000e+00f);
                
            }
            group->V[lid] = lV;
            group->RefracTime[lid] = lRefracTime;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 3) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord[threadIdx.x];
            }
        }
    }
    // merged2
    if(id >= 614400 && id < 921984) {
        unsigned int lo = 0;
        unsigned int hi = 5;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedNeuronUpdateGroupStartID2[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedNeuronUpdateGroup2 *group = &d_mergedNeuronUpdateGroup2[lo - 1]; 
        const unsigned int groupStartID = d_mergedNeuronUpdateGroupStartID2[lo - 1];
        const unsigned int lid = id - groupStartID;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar lRefracTime = group->RefracTime[lid];
            
            float Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn1[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if (lRefracTime <= 0.0f) {
              scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * group->Rmembrane) + (-6.50000000000000000e+01f);
              lV = alpha - (group->ExpTC * (alpha - lV));
            }
            else {
              lRefracTime -= DT;
            }
            
            // test for and register a true spike
            if (lRefracTime <= 0.0f && lV >= group->Vthresh) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord[threadIdx.x / 32], 1 << (threadIdx.x % 32));
                // spike reset code
                lV = (-6.50000000000000000e+01f);
                lRefracTime = (0.00000000000000000e+00f);
                
            }
            group->V[lid] = lV;
            group->RefracTime[lid] = lRefracTime;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 3) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord[threadIdx.x];
            }
        }
    }
}
void updateNeurons(float t, unsigned int recordingTimestep) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(96, 1);
        const dim3 grid(9604, 1);
        updateNeuronsKernel<<<grid, threads>>>(t, recordingTimestep);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
