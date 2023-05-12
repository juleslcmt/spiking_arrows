#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;
unsigned long long numRecordingTimesteps = 0;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntdown_neuron;
unsigned int* d_glbSpkCntdown_neuron;
unsigned int* glbSpkdown_neuron;
unsigned int* d_glbSpkdown_neuron;
uint32_t* recordSpkdown_neuron;
uint32_t* d_recordSpkdown_neuron;
scalar* Vdown_neuron;
scalar* d_Vdown_neuron;
scalar* RefracTimedown_neuron;
scalar* d_RefracTimedown_neuron;
unsigned int* glbSpkCntfilter_high_pop;
unsigned int* d_glbSpkCntfilter_high_pop;
unsigned int* glbSpkfilter_high_pop;
unsigned int* d_glbSpkfilter_high_pop;
uint32_t* recordSpkfilter_high_pop;
uint32_t* d_recordSpkfilter_high_pop;
scalar* Vfilter_high_pop;
scalar* d_Vfilter_high_pop;
scalar* RefracTimefilter_high_pop;
scalar* d_RefracTimefilter_high_pop;
unsigned int* glbSpkCntfilter_low_pop;
unsigned int* d_glbSpkCntfilter_low_pop;
unsigned int* glbSpkfilter_low_pop;
unsigned int* d_glbSpkfilter_low_pop;
uint32_t* recordSpkfilter_low_pop;
uint32_t* d_recordSpkfilter_low_pop;
scalar* Vfilter_low_pop;
scalar* d_Vfilter_low_pop;
scalar* RefracTimefilter_low_pop;
scalar* d_RefracTimefilter_low_pop;
unsigned int* glbSpkCntinput;
unsigned int* d_glbSpkCntinput;
unsigned int* glbSpkinput;
unsigned int* d_glbSpkinput;
uint32_t* recordSpkinput;
uint32_t* d_recordSpkinput;
uint32_t* inputinput;
uint32_t* d_inputinput;
unsigned int* glbSpkCntleft_neuron;
unsigned int* d_glbSpkCntleft_neuron;
unsigned int* glbSpkleft_neuron;
unsigned int* d_glbSpkleft_neuron;
uint32_t* recordSpkleft_neuron;
uint32_t* d_recordSpkleft_neuron;
scalar* Vleft_neuron;
scalar* d_Vleft_neuron;
scalar* RefracTimeleft_neuron;
scalar* d_RefracTimeleft_neuron;
unsigned int* glbSpkCntright_neuron;
unsigned int* d_glbSpkCntright_neuron;
unsigned int* glbSpkright_neuron;
unsigned int* d_glbSpkright_neuron;
uint32_t* recordSpkright_neuron;
uint32_t* d_recordSpkright_neuron;
scalar* Vright_neuron;
scalar* d_Vright_neuron;
scalar* RefracTimeright_neuron;
scalar* d_RefracTimeright_neuron;
unsigned int* glbSpkCntup_neuron;
unsigned int* d_glbSpkCntup_neuron;
unsigned int* glbSpkup_neuron;
unsigned int* d_glbSpkup_neuron;
uint32_t* recordSpkup_neuron;
uint32_t* d_recordSpkup_neuron;
scalar* Vup_neuron;
scalar* d_Vup_neuron;
scalar* RefracTimeup_neuron;
scalar* d_RefracTimeup_neuron;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSyninhibitory_down_neuron;
float* d_inSyninhibitory_down_neuron;
float* inSynexcitatory_down_neuron;
float* d_inSynexcitatory_down_neuron;
float* inSyninput_to_high_filter;
float* d_inSyninput_to_high_filter;
float* inSynhigh_to_low;
float* d_inSynhigh_to_low;
float* inSyninput_to_low_filter;
float* d_inSyninput_to_low_filter;
float* inSyninhibitory_left_neuron;
float* d_inSyninhibitory_left_neuron;
float* inSynexcitatory_left_neuron;
float* d_inSynexcitatory_left_neuron;
float* inSyninhibitory_right_neuron;
float* d_inSyninhibitory_right_neuron;
float* inSynexcitatory_right_neuron;
float* d_inSynexcitatory_right_neuron;
float* inSyninhibitory_up_neuron;
float* d_inSyninhibitory_up_neuron;
float* inSynexcitatory_up_neuron;
float* d_inSynexcitatory_up_neuron;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthhigh_to_low = 1;
unsigned int* rowLengthhigh_to_low;
unsigned int* d_rowLengthhigh_to_low;
uint32_t* indhigh_to_low;
uint32_t* d_indhigh_to_low;
const unsigned int maxRowLengthinput_to_high_filter = 1;
unsigned int* rowLengthinput_to_high_filter;
unsigned int* d_rowLengthinput_to_high_filter;
uint32_t* indinput_to_high_filter;
uint32_t* d_indinput_to_high_filter;
const unsigned int maxRowLengthinput_to_low_filter = 1;
unsigned int* rowLengthinput_to_low_filter;
unsigned int* d_rowLengthinput_to_low_filter;
uint32_t* indinput_to_low_filter;
uint32_t* d_indinput_to_low_filter;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* gexcitatory_down_neuron;
scalar* d_gexcitatory_down_neuron;
scalar* gexcitatory_left_neuron;
scalar* d_gexcitatory_left_neuron;
scalar* gexcitatory_right_neuron;
scalar* d_gexcitatory_right_neuron;
scalar* gexcitatory_up_neuron;
scalar* d_gexcitatory_up_neuron;
scalar* ghigh_to_low;
scalar* d_ghigh_to_low;
scalar* ginhibitory_down_neuron;
scalar* d_ginhibitory_down_neuron;
scalar* ginhibitory_left_neuron;
scalar* d_ginhibitory_left_neuron;
scalar* ginhibitory_right_neuron;
scalar* d_ginhibitory_right_neuron;
scalar* ginhibitory_up_neuron;
scalar* d_ginhibitory_up_neuron;
scalar* ginput_to_high_filter;
scalar* d_ginput_to_high_filter;
scalar* ginput_to_low_filter;
scalar* d_ginput_to_low_filter;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocateinputinput(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inputinput, count * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inputinput, count * sizeof(uint32_t)));
    pushMergedNeuronUpdate0inputToDevice(0, d_inputinput);
}
void freeinputinput() {
    CHECK_CUDA_ERRORS(cudaFreeHost(inputinput));
    CHECK_CUDA_ERRORS(cudaFree(d_inputinput));
}
void pushinputinputToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inputinput, inputinput, count * sizeof(uint32_t), cudaMemcpyHostToDevice));
}
void pullinputinputFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(inputinput, d_inputinput, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushdown_neuronSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntdown_neuron, glbSpkCntdown_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkdown_neuron, glbSpkdown_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushdown_neuronCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntdown_neuron, glbSpkCntdown_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkdown_neuron, glbSpkdown_neuron, glbSpkCntdown_neuron[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVdown_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vdown_neuron, Vdown_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVdown_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vdown_neuron, Vdown_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushRefracTimedown_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimedown_neuron, RefracTimedown_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentRefracTimedown_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimedown_neuron, RefracTimedown_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushdown_neuronStateToDevice(bool uninitialisedOnly) {
    pushVdown_neuronToDevice(uninitialisedOnly);
    pushRefracTimedown_neuronToDevice(uninitialisedOnly);
}

void pushfilter_high_popSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntfilter_high_pop, glbSpkCntfilter_high_pop, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkfilter_high_pop, glbSpkfilter_high_pop, 307200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushfilter_high_popCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntfilter_high_pop, glbSpkCntfilter_high_pop, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkfilter_high_pop, glbSpkfilter_high_pop, glbSpkCntfilter_high_pop[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVfilter_high_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vfilter_high_pop, Vfilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVfilter_high_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vfilter_high_pop, Vfilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushRefracTimefilter_high_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimefilter_high_pop, RefracTimefilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentRefracTimefilter_high_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimefilter_high_pop, RefracTimefilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushfilter_high_popStateToDevice(bool uninitialisedOnly) {
    pushVfilter_high_popToDevice(uninitialisedOnly);
    pushRefracTimefilter_high_popToDevice(uninitialisedOnly);
}

void pushfilter_low_popSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntfilter_low_pop, glbSpkCntfilter_low_pop, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkfilter_low_pop, glbSpkfilter_low_pop, 307200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushfilter_low_popCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntfilter_low_pop, glbSpkCntfilter_low_pop, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkfilter_low_pop, glbSpkfilter_low_pop, glbSpkCntfilter_low_pop[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVfilter_low_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vfilter_low_pop, Vfilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVfilter_low_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vfilter_low_pop, Vfilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushRefracTimefilter_low_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimefilter_low_pop, RefracTimefilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentRefracTimefilter_low_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimefilter_low_pop, RefracTimefilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushfilter_low_popStateToDevice(bool uninitialisedOnly) {
    pushVfilter_low_popToDevice(uninitialisedOnly);
    pushRefracTimefilter_low_popToDevice(uninitialisedOnly);
}

void pushinputSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntinput, glbSpkCntinput, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkinput, glbSpkinput, 307200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushinputCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntinput, glbSpkCntinput, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkinput, glbSpkinput, glbSpkCntinput[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushinputStateToDevice(bool uninitialisedOnly) {
}

void pushleft_neuronSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntleft_neuron, glbSpkCntleft_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkleft_neuron, glbSpkleft_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushleft_neuronCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntleft_neuron, glbSpkCntleft_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkleft_neuron, glbSpkleft_neuron, glbSpkCntleft_neuron[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVleft_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vleft_neuron, Vleft_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVleft_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vleft_neuron, Vleft_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushRefracTimeleft_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimeleft_neuron, RefracTimeleft_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentRefracTimeleft_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimeleft_neuron, RefracTimeleft_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushleft_neuronStateToDevice(bool uninitialisedOnly) {
    pushVleft_neuronToDevice(uninitialisedOnly);
    pushRefracTimeleft_neuronToDevice(uninitialisedOnly);
}

void pushright_neuronSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntright_neuron, glbSpkCntright_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkright_neuron, glbSpkright_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushright_neuronCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntright_neuron, glbSpkCntright_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkright_neuron, glbSpkright_neuron, glbSpkCntright_neuron[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVright_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vright_neuron, Vright_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVright_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vright_neuron, Vright_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushRefracTimeright_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimeright_neuron, RefracTimeright_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentRefracTimeright_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimeright_neuron, RefracTimeright_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushright_neuronStateToDevice(bool uninitialisedOnly) {
    pushVright_neuronToDevice(uninitialisedOnly);
    pushRefracTimeright_neuronToDevice(uninitialisedOnly);
}

void pushup_neuronSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntup_neuron, glbSpkCntup_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkup_neuron, glbSpkup_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushup_neuronCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntup_neuron, glbSpkCntup_neuron, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkup_neuron, glbSpkup_neuron, glbSpkCntup_neuron[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVup_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vup_neuron, Vup_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVup_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vup_neuron, Vup_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushRefracTimeup_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimeup_neuron, RefracTimeup_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentRefracTimeup_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_RefracTimeup_neuron, RefracTimeup_neuron, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushup_neuronStateToDevice(bool uninitialisedOnly) {
    pushVup_neuronToDevice(uninitialisedOnly);
    pushRefracTimeup_neuronToDevice(uninitialisedOnly);
}

void pushhigh_to_lowConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthhigh_to_low, rowLengthhigh_to_low, 307200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indhigh_to_low, indhigh_to_low, 307200 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushinput_to_high_filterConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthinput_to_high_filter, rowLengthinput_to_high_filter, 307200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indinput_to_high_filter, indinput_to_high_filter, 307200 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushinput_to_low_filterConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthinput_to_low_filter, rowLengthinput_to_low_filter, 307200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indinput_to_low_filter, indinput_to_low_filter, 307200 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushgexcitatory_down_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gexcitatory_down_neuron, gexcitatory_down_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSynexcitatory_down_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynexcitatory_down_neuron, inSynexcitatory_down_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushexcitatory_down_neuronStateToDevice(bool uninitialisedOnly) {
    pushgexcitatory_down_neuronToDevice(uninitialisedOnly);
    pushinSynexcitatory_down_neuronToDevice(uninitialisedOnly);
}

void pushgexcitatory_left_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gexcitatory_left_neuron, gexcitatory_left_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSynexcitatory_left_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynexcitatory_left_neuron, inSynexcitatory_left_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushexcitatory_left_neuronStateToDevice(bool uninitialisedOnly) {
    pushgexcitatory_left_neuronToDevice(uninitialisedOnly);
    pushinSynexcitatory_left_neuronToDevice(uninitialisedOnly);
}

void pushgexcitatory_right_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gexcitatory_right_neuron, gexcitatory_right_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSynexcitatory_right_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynexcitatory_right_neuron, inSynexcitatory_right_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushexcitatory_right_neuronStateToDevice(bool uninitialisedOnly) {
    pushgexcitatory_right_neuronToDevice(uninitialisedOnly);
    pushinSynexcitatory_right_neuronToDevice(uninitialisedOnly);
}

void pushgexcitatory_up_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gexcitatory_up_neuron, gexcitatory_up_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSynexcitatory_up_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynexcitatory_up_neuron, inSynexcitatory_up_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushexcitatory_up_neuronStateToDevice(bool uninitialisedOnly) {
    pushgexcitatory_up_neuronToDevice(uninitialisedOnly);
    pushinSynexcitatory_up_neuronToDevice(uninitialisedOnly);
}

void pushghigh_to_lowToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_ghigh_to_low, ghigh_to_low, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynhigh_to_lowToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynhigh_to_low, inSynhigh_to_low, 307200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushhigh_to_lowStateToDevice(bool uninitialisedOnly) {
    pushghigh_to_lowToDevice(uninitialisedOnly);
    pushinSynhigh_to_lowToDevice(uninitialisedOnly);
}

void pushginhibitory_down_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ginhibitory_down_neuron, ginhibitory_down_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSyninhibitory_down_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyninhibitory_down_neuron, inSyninhibitory_down_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushinhibitory_down_neuronStateToDevice(bool uninitialisedOnly) {
    pushginhibitory_down_neuronToDevice(uninitialisedOnly);
    pushinSyninhibitory_down_neuronToDevice(uninitialisedOnly);
}

void pushginhibitory_left_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ginhibitory_left_neuron, ginhibitory_left_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSyninhibitory_left_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyninhibitory_left_neuron, inSyninhibitory_left_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushinhibitory_left_neuronStateToDevice(bool uninitialisedOnly) {
    pushginhibitory_left_neuronToDevice(uninitialisedOnly);
    pushinSyninhibitory_left_neuronToDevice(uninitialisedOnly);
}

void pushginhibitory_right_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ginhibitory_right_neuron, ginhibitory_right_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSyninhibitory_right_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyninhibitory_right_neuron, inSyninhibitory_right_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushinhibitory_right_neuronStateToDevice(bool uninitialisedOnly) {
    pushginhibitory_right_neuronToDevice(uninitialisedOnly);
    pushinSyninhibitory_right_neuronToDevice(uninitialisedOnly);
}

void pushginhibitory_up_neuronToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ginhibitory_up_neuron, ginhibitory_up_neuron, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSyninhibitory_up_neuronToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyninhibitory_up_neuron, inSyninhibitory_up_neuron, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushinhibitory_up_neuronStateToDevice(bool uninitialisedOnly) {
    pushginhibitory_up_neuronToDevice(uninitialisedOnly);
    pushinSyninhibitory_up_neuronToDevice(uninitialisedOnly);
}

void pushginput_to_high_filterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_ginput_to_high_filter, ginput_to_high_filter, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSyninput_to_high_filterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyninput_to_high_filter, inSyninput_to_high_filter, 307200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushinput_to_high_filterStateToDevice(bool uninitialisedOnly) {
    pushginput_to_high_filterToDevice(uninitialisedOnly);
    pushinSyninput_to_high_filterToDevice(uninitialisedOnly);
}

void pushginput_to_low_filterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_ginput_to_low_filter, ginput_to_low_filter, 307200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSyninput_to_low_filterToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyninput_to_low_filter, inSyninput_to_low_filter, 307200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushinput_to_low_filterStateToDevice(bool uninitialisedOnly) {
    pushginput_to_low_filterToDevice(uninitialisedOnly);
    pushinSyninput_to_low_filterToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pulldown_neuronSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntdown_neuron, d_glbSpkCntdown_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkdown_neuron, d_glbSpkdown_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pulldown_neuronCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntdown_neuron, d_glbSpkCntdown_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkdown_neuron, d_glbSpkdown_neuron, glbSpkCntdown_neuron[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVdown_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vdown_neuron, d_Vdown_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVdown_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vdown_neuron, d_Vdown_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullRefracTimedown_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimedown_neuron, d_RefracTimedown_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentRefracTimedown_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimedown_neuron, d_RefracTimedown_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldown_neuronStateFromDevice() {
    pullVdown_neuronFromDevice();
    pullRefracTimedown_neuronFromDevice();
}

void pullfilter_high_popSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntfilter_high_pop, d_glbSpkCntfilter_high_pop, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkfilter_high_pop, d_glbSpkfilter_high_pop, 307200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullfilter_high_popCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntfilter_high_pop, d_glbSpkCntfilter_high_pop, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkfilter_high_pop, d_glbSpkfilter_high_pop, glbSpkCntfilter_high_pop[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVfilter_high_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vfilter_high_pop, d_Vfilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVfilter_high_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vfilter_high_pop, d_Vfilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullRefracTimefilter_high_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimefilter_high_pop, d_RefracTimefilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentRefracTimefilter_high_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimefilter_high_pop, d_RefracTimefilter_high_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullfilter_high_popStateFromDevice() {
    pullVfilter_high_popFromDevice();
    pullRefracTimefilter_high_popFromDevice();
}

void pullfilter_low_popSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntfilter_low_pop, d_glbSpkCntfilter_low_pop, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkfilter_low_pop, d_glbSpkfilter_low_pop, 307200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullfilter_low_popCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntfilter_low_pop, d_glbSpkCntfilter_low_pop, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkfilter_low_pop, d_glbSpkfilter_low_pop, glbSpkCntfilter_low_pop[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVfilter_low_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vfilter_low_pop, d_Vfilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVfilter_low_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vfilter_low_pop, d_Vfilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullRefracTimefilter_low_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimefilter_low_pop, d_RefracTimefilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentRefracTimefilter_low_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimefilter_low_pop, d_RefracTimefilter_low_pop, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullfilter_low_popStateFromDevice() {
    pullVfilter_low_popFromDevice();
    pullRefracTimefilter_low_popFromDevice();
}

void pullinputSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntinput, d_glbSpkCntinput, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkinput, d_glbSpkinput, 307200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullinputCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntinput, d_glbSpkCntinput, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkinput, d_glbSpkinput, glbSpkCntinput[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullinputStateFromDevice() {
}

void pullleft_neuronSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntleft_neuron, d_glbSpkCntleft_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkleft_neuron, d_glbSpkleft_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullleft_neuronCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntleft_neuron, d_glbSpkCntleft_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkleft_neuron, d_glbSpkleft_neuron, glbSpkCntleft_neuron[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVleft_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vleft_neuron, d_Vleft_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVleft_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vleft_neuron, d_Vleft_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullRefracTimeleft_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimeleft_neuron, d_RefracTimeleft_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentRefracTimeleft_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimeleft_neuron, d_RefracTimeleft_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullleft_neuronStateFromDevice() {
    pullVleft_neuronFromDevice();
    pullRefracTimeleft_neuronFromDevice();
}

void pullright_neuronSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntright_neuron, d_glbSpkCntright_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkright_neuron, d_glbSpkright_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullright_neuronCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntright_neuron, d_glbSpkCntright_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkright_neuron, d_glbSpkright_neuron, glbSpkCntright_neuron[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVright_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vright_neuron, d_Vright_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVright_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vright_neuron, d_Vright_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullRefracTimeright_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimeright_neuron, d_RefracTimeright_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentRefracTimeright_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimeright_neuron, d_RefracTimeright_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullright_neuronStateFromDevice() {
    pullVright_neuronFromDevice();
    pullRefracTimeright_neuronFromDevice();
}

void pullup_neuronSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntup_neuron, d_glbSpkCntup_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkup_neuron, d_glbSpkup_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullup_neuronCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntup_neuron, d_glbSpkCntup_neuron, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkup_neuron, d_glbSpkup_neuron, glbSpkCntup_neuron[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVup_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vup_neuron, d_Vup_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVup_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vup_neuron, d_Vup_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullRefracTimeup_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimeup_neuron, d_RefracTimeup_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentRefracTimeup_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(RefracTimeup_neuron, d_RefracTimeup_neuron, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullup_neuronStateFromDevice() {
    pullVup_neuronFromDevice();
    pullRefracTimeup_neuronFromDevice();
}

void pullhigh_to_lowConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthhigh_to_low, d_rowLengthhigh_to_low, 307200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indhigh_to_low, d_indhigh_to_low, 307200 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullinput_to_high_filterConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthinput_to_high_filter, d_rowLengthinput_to_high_filter, 307200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indinput_to_high_filter, d_indinput_to_high_filter, 307200 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullinput_to_low_filterConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthinput_to_low_filter, d_rowLengthinput_to_low_filter, 307200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indinput_to_low_filter, d_indinput_to_low_filter, 307200 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullgexcitatory_down_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gexcitatory_down_neuron, d_gexcitatory_down_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynexcitatory_down_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynexcitatory_down_neuron, d_inSynexcitatory_down_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullexcitatory_down_neuronStateFromDevice() {
    pullgexcitatory_down_neuronFromDevice();
    pullinSynexcitatory_down_neuronFromDevice();
}

void pullgexcitatory_left_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gexcitatory_left_neuron, d_gexcitatory_left_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynexcitatory_left_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynexcitatory_left_neuron, d_inSynexcitatory_left_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullexcitatory_left_neuronStateFromDevice() {
    pullgexcitatory_left_neuronFromDevice();
    pullinSynexcitatory_left_neuronFromDevice();
}

void pullgexcitatory_right_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gexcitatory_right_neuron, d_gexcitatory_right_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynexcitatory_right_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynexcitatory_right_neuron, d_inSynexcitatory_right_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullexcitatory_right_neuronStateFromDevice() {
    pullgexcitatory_right_neuronFromDevice();
    pullinSynexcitatory_right_neuronFromDevice();
}

void pullgexcitatory_up_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gexcitatory_up_neuron, d_gexcitatory_up_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynexcitatory_up_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynexcitatory_up_neuron, d_inSynexcitatory_up_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullexcitatory_up_neuronStateFromDevice() {
    pullgexcitatory_up_neuronFromDevice();
    pullinSynexcitatory_up_neuronFromDevice();
}

void pullghigh_to_lowFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ghigh_to_low, d_ghigh_to_low, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynhigh_to_lowFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynhigh_to_low, d_inSynhigh_to_low, 307200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullhigh_to_lowStateFromDevice() {
    pullghigh_to_lowFromDevice();
    pullinSynhigh_to_lowFromDevice();
}

void pullginhibitory_down_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ginhibitory_down_neuron, d_ginhibitory_down_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSyninhibitory_down_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSyninhibitory_down_neuron, d_inSyninhibitory_down_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullinhibitory_down_neuronStateFromDevice() {
    pullginhibitory_down_neuronFromDevice();
    pullinSyninhibitory_down_neuronFromDevice();
}

void pullginhibitory_left_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ginhibitory_left_neuron, d_ginhibitory_left_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSyninhibitory_left_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSyninhibitory_left_neuron, d_inSyninhibitory_left_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullinhibitory_left_neuronStateFromDevice() {
    pullginhibitory_left_neuronFromDevice();
    pullinSyninhibitory_left_neuronFromDevice();
}

void pullginhibitory_right_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ginhibitory_right_neuron, d_ginhibitory_right_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSyninhibitory_right_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSyninhibitory_right_neuron, d_inSyninhibitory_right_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullinhibitory_right_neuronStateFromDevice() {
    pullginhibitory_right_neuronFromDevice();
    pullinSyninhibitory_right_neuronFromDevice();
}

void pullginhibitory_up_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ginhibitory_up_neuron, d_ginhibitory_up_neuron, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSyninhibitory_up_neuronFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSyninhibitory_up_neuron, d_inSyninhibitory_up_neuron, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullinhibitory_up_neuronStateFromDevice() {
    pullginhibitory_up_neuronFromDevice();
    pullinSyninhibitory_up_neuronFromDevice();
}

void pullginput_to_high_filterFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ginput_to_high_filter, d_ginput_to_high_filter, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSyninput_to_high_filterFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSyninput_to_high_filter, d_inSyninput_to_high_filter, 307200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullinput_to_high_filterStateFromDevice() {
    pullginput_to_high_filterFromDevice();
    pullinSyninput_to_high_filterFromDevice();
}

void pullginput_to_low_filterFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ginput_to_low_filter, d_ginput_to_low_filter, 307200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSyninput_to_low_filterFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSyninput_to_low_filter, d_inSyninput_to_low_filter, 307200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullinput_to_low_filterStateFromDevice() {
    pullginput_to_low_filterFromDevice();
    pullinSyninput_to_low_filterFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getdown_neuronCurrentSpikes(unsigned int batch) {
    return (glbSpkdown_neuron);
}

unsigned int& getdown_neuronCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntdown_neuron[0];
}

scalar* getCurrentVdown_neuron(unsigned int batch) {
    return Vdown_neuron;
}

scalar* getCurrentRefracTimedown_neuron(unsigned int batch) {
    return RefracTimedown_neuron;
}

unsigned int* getfilter_high_popCurrentSpikes(unsigned int batch) {
    return (glbSpkfilter_high_pop);
}

unsigned int& getfilter_high_popCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntfilter_high_pop[0];
}

scalar* getCurrentVfilter_high_pop(unsigned int batch) {
    return Vfilter_high_pop;
}

scalar* getCurrentRefracTimefilter_high_pop(unsigned int batch) {
    return RefracTimefilter_high_pop;
}

unsigned int* getfilter_low_popCurrentSpikes(unsigned int batch) {
    return (glbSpkfilter_low_pop);
}

unsigned int& getfilter_low_popCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntfilter_low_pop[0];
}

scalar* getCurrentVfilter_low_pop(unsigned int batch) {
    return Vfilter_low_pop;
}

scalar* getCurrentRefracTimefilter_low_pop(unsigned int batch) {
    return RefracTimefilter_low_pop;
}

unsigned int* getinputCurrentSpikes(unsigned int batch) {
    return (glbSpkinput);
}

unsigned int& getinputCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntinput[0];
}

unsigned int* getleft_neuronCurrentSpikes(unsigned int batch) {
    return (glbSpkleft_neuron);
}

unsigned int& getleft_neuronCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntleft_neuron[0];
}

scalar* getCurrentVleft_neuron(unsigned int batch) {
    return Vleft_neuron;
}

scalar* getCurrentRefracTimeleft_neuron(unsigned int batch) {
    return RefracTimeleft_neuron;
}

unsigned int* getright_neuronCurrentSpikes(unsigned int batch) {
    return (glbSpkright_neuron);
}

unsigned int& getright_neuronCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntright_neuron[0];
}

scalar* getCurrentVright_neuron(unsigned int batch) {
    return Vright_neuron;
}

scalar* getCurrentRefracTimeright_neuron(unsigned int batch) {
    return RefracTimeright_neuron;
}

unsigned int* getup_neuronCurrentSpikes(unsigned int batch) {
    return (glbSpkup_neuron);
}

unsigned int& getup_neuronCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntup_neuron[0];
}

scalar* getCurrentVup_neuron(unsigned int batch) {
    return Vup_neuron;
}

scalar* getCurrentRefracTimeup_neuron(unsigned int batch) {
    return RefracTimeup_neuron;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushdown_neuronStateToDevice(uninitialisedOnly);
    pushfilter_high_popStateToDevice(uninitialisedOnly);
    pushfilter_low_popStateToDevice(uninitialisedOnly);
    pushinputStateToDevice(uninitialisedOnly);
    pushleft_neuronStateToDevice(uninitialisedOnly);
    pushright_neuronStateToDevice(uninitialisedOnly);
    pushup_neuronStateToDevice(uninitialisedOnly);
    pushexcitatory_down_neuronStateToDevice(uninitialisedOnly);
    pushexcitatory_left_neuronStateToDevice(uninitialisedOnly);
    pushexcitatory_right_neuronStateToDevice(uninitialisedOnly);
    pushexcitatory_up_neuronStateToDevice(uninitialisedOnly);
    pushhigh_to_lowStateToDevice(uninitialisedOnly);
    pushinhibitory_down_neuronStateToDevice(uninitialisedOnly);
    pushinhibitory_left_neuronStateToDevice(uninitialisedOnly);
    pushinhibitory_right_neuronStateToDevice(uninitialisedOnly);
    pushinhibitory_up_neuronStateToDevice(uninitialisedOnly);
    pushinput_to_high_filterStateToDevice(uninitialisedOnly);
    pushinput_to_low_filterStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushhigh_to_lowConnectivityToDevice(uninitialisedOnly);
    pushinput_to_high_filterConnectivityToDevice(uninitialisedOnly);
    pushinput_to_low_filterConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pulldown_neuronStateFromDevice();
    pullfilter_high_popStateFromDevice();
    pullfilter_low_popStateFromDevice();
    pullinputStateFromDevice();
    pullleft_neuronStateFromDevice();
    pullright_neuronStateFromDevice();
    pullup_neuronStateFromDevice();
    pullexcitatory_down_neuronStateFromDevice();
    pullexcitatory_left_neuronStateFromDevice();
    pullexcitatory_right_neuronStateFromDevice();
    pullexcitatory_up_neuronStateFromDevice();
    pullhigh_to_lowStateFromDevice();
    pullinhibitory_down_neuronStateFromDevice();
    pullinhibitory_left_neuronStateFromDevice();
    pullinhibitory_right_neuronStateFromDevice();
    pullinhibitory_up_neuronStateFromDevice();
    pullinput_to_high_filterStateFromDevice();
    pullinput_to_low_filterStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pulldown_neuronCurrentSpikesFromDevice();
    pullfilter_high_popCurrentSpikesFromDevice();
    pullfilter_low_popCurrentSpikesFromDevice();
    pullinputCurrentSpikesFromDevice();
    pullleft_neuronCurrentSpikesFromDevice();
    pullright_neuronCurrentSpikesFromDevice();
    pullup_neuronCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
        const unsigned int numWords = 1 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkdown_neuron, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkdown_neuron, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate2recordSpkToDevice(0, d_recordSpkdown_neuron);
        }
    }
     {
        const unsigned int numWords = 9600 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkfilter_high_pop, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkfilter_high_pop, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate1recordSpkToDevice(0, d_recordSpkfilter_high_pop);
        }
    }
     {
        const unsigned int numWords = 9600 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkfilter_low_pop, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkfilter_low_pop, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate2recordSpkToDevice(1, d_recordSpkfilter_low_pop);
        }
    }
     {
        const unsigned int numWords = 9600 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkinput, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkinput, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate0recordSpkToDevice(0, d_recordSpkinput);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkleft_neuron, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkleft_neuron, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate2recordSpkToDevice(2, d_recordSpkleft_neuron);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkright_neuron, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkright_neuron, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate2recordSpkToDevice(3, d_recordSpkright_neuron);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkup_neuron, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkup_neuron, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate2recordSpkToDevice(4, d_recordSpkup_neuron);
        }
    }
}

void pullRecordingBuffersFromDevice() {
    if(numRecordingTimesteps == 0) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkdown_neuron, d_recordSpkdown_neuron, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 9600 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkfilter_high_pop, d_recordSpkfilter_high_pop, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 9600 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkfilter_low_pop, d_recordSpkfilter_low_pop, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 9600 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkinput, d_recordSpkinput, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkleft_neuron, d_recordSpkleft_neuron, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkright_neuron, d_recordSpkright_neuron, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkup_neuron, d_recordSpkup_neuron, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:01:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntdown_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntdown_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkdown_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkdown_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vdown_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vdown_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&RefracTimedown_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_RefracTimedown_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntfilter_high_pop, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntfilter_high_pop, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkfilter_high_pop, 307200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkfilter_high_pop, 307200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vfilter_high_pop, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vfilter_high_pop, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&RefracTimefilter_high_pop, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_RefracTimefilter_high_pop, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntfilter_low_pop, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntfilter_low_pop, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkfilter_low_pop, 307200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkfilter_low_pop, 307200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vfilter_low_pop, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vfilter_low_pop, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&RefracTimefilter_low_pop, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_RefracTimefilter_low_pop, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntinput, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntinput, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkinput, 307200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkinput, 307200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntleft_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntleft_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkleft_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkleft_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vleft_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vleft_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&RefracTimeleft_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_RefracTimeleft_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntright_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntright_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkright_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkright_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vright_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vright_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&RefracTimeright_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_RefracTimeright_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntup_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntup_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkup_neuron, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkup_neuron, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vup_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vup_neuron, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&RefracTimeup_neuron, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_RefracTimeup_neuron, 1 * sizeof(scalar)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSyninhibitory_down_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSyninhibitory_down_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynexcitatory_down_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynexcitatory_down_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSyninput_to_high_filter, 307200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSyninput_to_high_filter, 307200 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynhigh_to_low, 307200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynhigh_to_low, 307200 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSyninput_to_low_filter, 307200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSyninput_to_low_filter, 307200 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSyninhibitory_left_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSyninhibitory_left_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynexcitatory_left_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynexcitatory_left_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSyninhibitory_right_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSyninhibitory_right_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynexcitatory_right_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynexcitatory_right_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSyninhibitory_up_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSyninhibitory_up_neuron, 1 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynexcitatory_up_neuron, 1 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynexcitatory_up_neuron, 1 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthhigh_to_low, 307200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthhigh_to_low, 307200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indhigh_to_low, 307200 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indhigh_to_low, 307200 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthinput_to_high_filter, 307200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthinput_to_high_filter, 307200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indinput_to_high_filter, 307200 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indinput_to_high_filter, 307200 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthinput_to_low_filter, 307200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthinput_to_low_filter, 307200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indinput_to_low_filter, 307200 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indinput_to_low_filter, 307200 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gexcitatory_down_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gexcitatory_down_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gexcitatory_left_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gexcitatory_left_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gexcitatory_right_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gexcitatory_right_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gexcitatory_up_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gexcitatory_up_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ghigh_to_low, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ghigh_to_low, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ginhibitory_down_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ginhibitory_down_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ginhibitory_left_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ginhibitory_left_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ginhibitory_right_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ginhibitory_right_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ginhibitory_up_neuron, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ginhibitory_up_neuron, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ginput_to_high_filter, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ginput_to_high_filter, 307200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ginput_to_low_filter, 307200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ginput_to_low_filter, 307200 * sizeof(scalar)));
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntinput, d_glbSpkinput, 307200);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_Vfilter_high_pop, d_RefracTimefilter_high_pop, d_inSyninput_to_high_filter, 307200);
    pushMergedNeuronInitGroup2ToDevice(0, d_glbSpkCntdown_neuron, d_glbSpkdown_neuron, d_Vdown_neuron, d_RefracTimedown_neuron, d_inSyninhibitory_down_neuron, d_inSynexcitatory_down_neuron, 1);
    pushMergedNeuronInitGroup2ToDevice(1, d_glbSpkCntfilter_low_pop, d_glbSpkfilter_low_pop, d_Vfilter_low_pop, d_RefracTimefilter_low_pop, d_inSynhigh_to_low, d_inSyninput_to_low_filter, 307200);
    pushMergedNeuronInitGroup2ToDevice(2, d_glbSpkCntleft_neuron, d_glbSpkleft_neuron, d_Vleft_neuron, d_RefracTimeleft_neuron, d_inSyninhibitory_left_neuron, d_inSynexcitatory_left_neuron, 1);
    pushMergedNeuronInitGroup2ToDevice(3, d_glbSpkCntright_neuron, d_glbSpkright_neuron, d_Vright_neuron, d_RefracTimeright_neuron, d_inSyninhibitory_right_neuron, d_inSynexcitatory_right_neuron, 1);
    pushMergedNeuronInitGroup2ToDevice(4, d_glbSpkCntup_neuron, d_glbSpkup_neuron, d_Vup_neuron, d_RefracTimeup_neuron, d_inSyninhibitory_up_neuron, d_inSynexcitatory_up_neuron, 1);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthhigh_to_low, d_indhigh_to_low, 1, 307200, 307200);
    pushMergedSynapseConnectivityInitGroup0ToDevice(1, d_rowLengthinput_to_high_filter, d_indinput_to_high_filter, 1, 307200, 307200);
    pushMergedSynapseConnectivityInitGroup0ToDevice(2, d_rowLengthinput_to_low_filter, d_indinput_to_low_filter, 1, 307200, 307200);
    pushMergedSynapseSparseInitGroup0ToDevice(0, d_rowLengthhigh_to_low, d_indhigh_to_low, d_ghigh_to_low, 1, 307200, 307200, 1, -1.40000000000000000e+03f);
    pushMergedSynapseSparseInitGroup0ToDevice(1, d_rowLengthinput_to_high_filter, d_indinput_to_high_filter, d_ginput_to_high_filter, 1, 307200, 307200, 1, 7.00000000000000000e+01f);
    pushMergedSynapseSparseInitGroup0ToDevice(2, d_rowLengthinput_to_low_filter, d_indinput_to_low_filter, d_ginput_to_low_filter, 1, 307200, 307200, 1, 7.00000000000000000e+01f);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntinput, d_glbSpkinput, d_inputinput, d_recordSpkinput, 307200);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_Vfilter_high_pop, d_RefracTimefilter_high_pop, d_inSyninput_to_high_filter, d_recordSpkfilter_high_pop, 307200);
    pushMergedNeuronUpdateGroup2ToDevice(0, d_glbSpkCntdown_neuron, d_glbSpkdown_neuron, d_Vdown_neuron, d_RefracTimedown_neuron, d_inSyninhibitory_down_neuron, d_inSynexcitatory_down_neuron, d_recordSpkdown_neuron, 1, 5.00000000000000000e-01f, -5.45000000000000000e+01f, 8.18730753077981821e-01f, 5.00000000000000000e-01f);
    pushMergedNeuronUpdateGroup2ToDevice(1, d_glbSpkCntfilter_low_pop, d_glbSpkfilter_low_pop, d_Vfilter_low_pop, d_RefracTimefilter_low_pop, d_inSynhigh_to_low, d_inSyninput_to_low_filter, d_recordSpkfilter_low_pop, 307200, 1.00000000000000000e+01f, -5.95000000000000000e+01f, 9.90049833749168107e-01f, 1.00000000000000000e+01f);
    pushMergedNeuronUpdateGroup2ToDevice(2, d_glbSpkCntleft_neuron, d_glbSpkleft_neuron, d_Vleft_neuron, d_RefracTimeleft_neuron, d_inSyninhibitory_left_neuron, d_inSynexcitatory_left_neuron, d_recordSpkleft_neuron, 1, 5.00000000000000000e-01f, -5.45000000000000000e+01f, 8.18730753077981821e-01f, 5.00000000000000000e-01f);
    pushMergedNeuronUpdateGroup2ToDevice(3, d_glbSpkCntright_neuron, d_glbSpkright_neuron, d_Vright_neuron, d_RefracTimeright_neuron, d_inSyninhibitory_right_neuron, d_inSynexcitatory_right_neuron, d_recordSpkright_neuron, 1, 5.00000000000000000e-01f, -5.45000000000000000e+01f, 8.18730753077981821e-01f, 5.00000000000000000e-01f);
    pushMergedNeuronUpdateGroup2ToDevice(4, d_glbSpkCntup_neuron, d_glbSpkup_neuron, d_Vup_neuron, d_RefracTimeup_neuron, d_inSyninhibitory_up_neuron, d_inSynexcitatory_up_neuron, d_recordSpkup_neuron, 1, 5.00000000000000000e-01f, -5.45000000000000000e+01f, 8.18730753077981821e-01f, 5.00000000000000000e-01f);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynhigh_to_low, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_rowLengthhigh_to_low, d_indhigh_to_low, d_ghigh_to_low, 1, 307200, 307200);
    pushMergedPresynapticUpdateGroup0ToDevice(1, d_inSyninput_to_high_filter, d_glbSpkCntinput, d_glbSpkinput, d_rowLengthinput_to_high_filter, d_indinput_to_high_filter, d_ginput_to_high_filter, 1, 307200, 307200);
    pushMergedPresynapticUpdateGroup0ToDevice(2, d_inSyninput_to_low_filter, d_glbSpkCntinput, d_glbSpkinput, d_rowLengthinput_to_low_filter, d_indinput_to_low_filter, d_ginput_to_low_filter, 1, 307200, 307200);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynexcitatory_down_neuron, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_gexcitatory_down_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(1, d_inSynexcitatory_left_neuron, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_gexcitatory_left_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(2, d_inSynexcitatory_right_neuron, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_gexcitatory_right_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(3, d_inSynexcitatory_up_neuron, d_glbSpkCntfilter_high_pop, d_glbSpkfilter_high_pop, d_gexcitatory_up_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(4, d_inSyninhibitory_down_neuron, d_glbSpkCntfilter_low_pop, d_glbSpkfilter_low_pop, d_ginhibitory_down_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(5, d_inSyninhibitory_left_neuron, d_glbSpkCntfilter_low_pop, d_glbSpkfilter_low_pop, d_ginhibitory_left_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(6, d_inSyninhibitory_right_neuron, d_glbSpkCntfilter_low_pop, d_glbSpkfilter_low_pop, d_ginhibitory_right_neuron, 1, 307200, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(7, d_inSyninhibitory_up_neuron, d_glbSpkCntfilter_low_pop, d_glbSpkfilter_low_pop, d_ginhibitory_up_neuron, 1, 307200, 1);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntfilter_high_pop);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(1, d_glbSpkCntfilter_low_pop);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(2, d_glbSpkCntinput);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(0, d_glbSpkCntdown_neuron);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(1, d_glbSpkCntleft_neuron);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(2, d_glbSpkCntright_neuron);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(3, d_glbSpkCntup_neuron);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntdown_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntdown_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkdown_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkdown_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkdown_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkdown_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vdown_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_Vdown_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(RefracTimedown_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_RefracTimedown_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_Vfilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(RefracTimefilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_RefracTimefilter_high_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_Vfilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(RefracTimefilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_RefracTimefilter_low_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntinput));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntinput));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkinput));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkinput));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkinput));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkinput));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntleft_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntleft_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkleft_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkleft_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkleft_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkleft_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vleft_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_Vleft_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(RefracTimeleft_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_RefracTimeleft_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntright_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntright_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkright_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkright_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkright_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkright_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vright_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_Vright_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(RefracTimeright_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_RefracTimeright_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntup_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntup_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkup_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkup_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkup_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkup_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vup_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_Vup_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(RefracTimeup_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_RefracTimeup_neuron));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSyninhibitory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSyninhibitory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynexcitatory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynexcitatory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSyninput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_inSyninput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynhigh_to_low));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynhigh_to_low));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSyninput_to_low_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_inSyninput_to_low_filter));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSyninhibitory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSyninhibitory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynexcitatory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynexcitatory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSyninhibitory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSyninhibitory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynexcitatory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynexcitatory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSyninhibitory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSyninhibitory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynexcitatory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynexcitatory_up_neuron));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthhigh_to_low));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthhigh_to_low));
    CHECK_CUDA_ERRORS(cudaFreeHost(indhigh_to_low));
    CHECK_CUDA_ERRORS(cudaFree(d_indhigh_to_low));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthinput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthinput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFreeHost(indinput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_indinput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthinput_to_low_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthinput_to_low_filter));
    CHECK_CUDA_ERRORS(cudaFreeHost(indinput_to_low_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_indinput_to_low_filter));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(gexcitatory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_gexcitatory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(gexcitatory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_gexcitatory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(gexcitatory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_gexcitatory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(gexcitatory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_gexcitatory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(ghigh_to_low));
    CHECK_CUDA_ERRORS(cudaFree(d_ghigh_to_low));
    CHECK_CUDA_ERRORS(cudaFreeHost(ginhibitory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_ginhibitory_down_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(ginhibitory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_ginhibitory_left_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(ginhibitory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_ginhibitory_right_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(ginhibitory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFree(d_ginhibitory_up_neuron));
    CHECK_CUDA_ERRORS(cudaFreeHost(ginput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_ginput_to_high_filter));
    CHECK_CUDA_ERRORS(cudaFreeHost(ginput_to_low_filter));
    CHECK_CUDA_ERRORS(cudaFree(d_ginput_to_low_filter));
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

