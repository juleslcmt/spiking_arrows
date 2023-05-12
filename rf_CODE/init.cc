#include "definitionsInternal.h"
struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* U;
    float* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* startSpike;
    unsigned int* endSpike;
    unsigned int numNeurons;
    
}
;
static MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* U, float* inSynInSyn0, unsigned int numNeurons) {
    mergedNeuronInitGroup0[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup0[idx].spk = spk;
    mergedNeuronInitGroup0[idx].V = V;
    mergedNeuronInitGroup0[idx].U = U;
    mergedNeuronInitGroup0[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup0[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup1 mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* startSpike, unsigned int* endSpike, unsigned int numNeurons) {
    mergedNeuronInitGroup1[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup1[idx].spk = spk;
    mergedNeuronInitGroup1[idx].startSpike = startSpike;
    mergedNeuronInitGroup1[idx].endSpike = endSpike;
    mergedNeuronInitGroup1[idx].numNeurons = numNeurons;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void initialize() {
    // ------------------------------------------------------------------------
    // Neuron groups
     {
        // merged neuron init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup0[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->U[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000000e+00f;
                }
            }
            // current source variables
        }
    }
     {
        // merged neuron init group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup1[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    unsigned int initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->startSpike[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    unsigned int initVal;
                    initVal = (3.00000000000000000e+01f);
                    group->endSpike[i] = initVal;
                }
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
    // Synapse sparse connectivity
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // ------------------------------------------------------------------------
    // Custom sparse WU update groups
}
