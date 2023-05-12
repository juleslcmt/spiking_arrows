#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* U;
    float* inSynInSyn0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* startSpike;
    unsigned int* endSpike;
    scalar* spikeTimes;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
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
static MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* U, float* inSynInSyn0, uint32_t* recordSpk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup0[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup0[idx].spk = spk;
    mergedNeuronUpdateGroup0[idx].V = V;
    mergedNeuronUpdateGroup0[idx].U = U;
    mergedNeuronUpdateGroup0[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup0[idx].recordSpk = recordSpk;
    mergedNeuronUpdateGroup0[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* startSpike, unsigned int* endSpike, scalar* spikeTimes, uint32_t* recordSpk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup1[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup1[idx].spk = spk;
    mergedNeuronUpdateGroup1[idx].startSpike = startSpike;
    mergedNeuronUpdateGroup1[idx].endSpike = endSpike;
    mergedNeuronUpdateGroup1[idx].spikeTimes = spikeTimes;
    mergedNeuronUpdateGroup1[idx].recordSpk = recordSpk;
    mergedNeuronUpdateGroup1[idx].numNeurons = numNeurons;
}
static MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[1];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    mergedNeuronSpikeQueueUpdateGroup0[idx].spkCnt = spkCnt;
}
static MergedNeuronSpikeQueueUpdateGroup1 mergedNeuronSpikeQueueUpdateGroup1[1];
void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt) {
    mergedNeuronSpikeQueueUpdateGroup1[idx].spkCnt = spkCnt;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value) {
    mergedNeuronUpdateGroup0[idx].recordSpk = value;
}

void pushMergedNeuronUpdate1spikeTimesToDevice(unsigned int idx, scalar* value) {
    mergedNeuronUpdateGroup1[idx].spikeTimes = value;
}

void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value) {
    mergedNeuronUpdateGroup1[idx].recordSpk = value;
}

void updateNeurons(float t, unsigned int recordingTimestep) {
     {
        // merged neuron spike queue update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronSpikeQueueUpdateGroup0[g]; 
            group->spkCnt[0] = 0;
        }
    }
     {
        // merged neuron spike queue update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronSpikeQueueUpdateGroup1[g]; 
            group->spkCnt[0] = 0;
        }
    }
     {
        // merged neuron update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup0[g]; 
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                scalar lV = group->V[i];
                scalar lU = group->U[i];
                
                float Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn0[i];
                    Isyn += linSyn; linSyn = 0;
                    
                    group->inSynInSyn0[i] = linSyn;
                }
                // test whether spike condition was fulfilled previously
                const bool oldSpike = (
                lV >= 1.0f
                );
                // calculate membrane potential
                
                const scalar oldV = lV;
                const scalar oldU = lU;
                lV += DT * oldU;
                lU += DT * (Isyn - (2.0f * (2.50000000000000000e+00f) * oldU) - ((6.91150383789754574e+00f) * (6.91150383789754574e+00f) * oldV));
                
                
                // test for and register a true spike
                if ((
                lV >= 1.0f
                ) && !(oldSpike)) {
                    group->spk[group->spkCnt[0]++] = i;
                    group->recordSpk[(recordingTimestep * numRecordingWords) + (i / 32)] |= (1 << (i % 32));
                    // spike reset code
                    
                    
                }
                group->V[i] = lV;
                group->U[i] = lU;
            }
        }
    }
     {
        // merged neuron update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup1[g]; 
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                unsigned int lstartSpike = group->startSpike[i];
                const unsigned int lendSpike = group->endSpike[i];
                
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                
                // test for and register a true spike
                if (lstartSpike != lendSpike && t >= group->spikeTimes[lstartSpike]) {
                    group->spk[group->spkCnt[0]++] = i;
                    group->recordSpk[(recordingTimestep * numRecordingWords) + (i / 32)] |= (1 << (i % 32));
                    // spike reset code
                    lstartSpike++;
                    
                }
                group->startSpike[i] = lstartSpike;
            }
        }
    }
}
