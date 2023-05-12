#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cstdint>
#define DT 1.00000000000000006e-01f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_down_neuron glbSpkCntdown_neuron[0]
#define spike_down_neuron glbSpkdown_neuron
#define glbSpkShiftdown_neuron 0

EXPORT_VAR unsigned int* glbSpkCntdown_neuron;
EXPORT_VAR unsigned int* d_glbSpkCntdown_neuron;
EXPORT_VAR unsigned int* glbSpkdown_neuron;
EXPORT_VAR unsigned int* d_glbSpkdown_neuron;
EXPORT_VAR uint32_t* recordSpkdown_neuron;
EXPORT_VAR uint32_t* d_recordSpkdown_neuron;
EXPORT_VAR scalar* Vdown_neuron;
EXPORT_VAR scalar* d_Vdown_neuron;
EXPORT_VAR scalar* RefracTimedown_neuron;
EXPORT_VAR scalar* d_RefracTimedown_neuron;
#define spikeCount_filter_high_pop glbSpkCntfilter_high_pop[0]
#define spike_filter_high_pop glbSpkfilter_high_pop
#define glbSpkShiftfilter_high_pop 0

EXPORT_VAR unsigned int* glbSpkCntfilter_high_pop;
EXPORT_VAR unsigned int* d_glbSpkCntfilter_high_pop;
EXPORT_VAR unsigned int* glbSpkfilter_high_pop;
EXPORT_VAR unsigned int* d_glbSpkfilter_high_pop;
EXPORT_VAR uint32_t* recordSpkfilter_high_pop;
EXPORT_VAR uint32_t* d_recordSpkfilter_high_pop;
EXPORT_VAR scalar* Vfilter_high_pop;
EXPORT_VAR scalar* d_Vfilter_high_pop;
EXPORT_VAR scalar* RefracTimefilter_high_pop;
EXPORT_VAR scalar* d_RefracTimefilter_high_pop;
#define spikeCount_filter_low_pop glbSpkCntfilter_low_pop[0]
#define spike_filter_low_pop glbSpkfilter_low_pop
#define glbSpkShiftfilter_low_pop 0

EXPORT_VAR unsigned int* glbSpkCntfilter_low_pop;
EXPORT_VAR unsigned int* d_glbSpkCntfilter_low_pop;
EXPORT_VAR unsigned int* glbSpkfilter_low_pop;
EXPORT_VAR unsigned int* d_glbSpkfilter_low_pop;
EXPORT_VAR uint32_t* recordSpkfilter_low_pop;
EXPORT_VAR uint32_t* d_recordSpkfilter_low_pop;
EXPORT_VAR scalar* Vfilter_low_pop;
EXPORT_VAR scalar* d_Vfilter_low_pop;
EXPORT_VAR scalar* RefracTimefilter_low_pop;
EXPORT_VAR scalar* d_RefracTimefilter_low_pop;
#define spikeCount_input glbSpkCntinput[0]
#define spike_input glbSpkinput
#define glbSpkShiftinput 0

EXPORT_VAR unsigned int* glbSpkCntinput;
EXPORT_VAR unsigned int* d_glbSpkCntinput;
EXPORT_VAR unsigned int* glbSpkinput;
EXPORT_VAR unsigned int* d_glbSpkinput;
EXPORT_VAR uint32_t* recordSpkinput;
EXPORT_VAR uint32_t* d_recordSpkinput;
EXPORT_VAR uint32_t* inputinput;
EXPORT_VAR uint32_t* d_inputinput;
#define spikeCount_left_neuron glbSpkCntleft_neuron[0]
#define spike_left_neuron glbSpkleft_neuron
#define glbSpkShiftleft_neuron 0

EXPORT_VAR unsigned int* glbSpkCntleft_neuron;
EXPORT_VAR unsigned int* d_glbSpkCntleft_neuron;
EXPORT_VAR unsigned int* glbSpkleft_neuron;
EXPORT_VAR unsigned int* d_glbSpkleft_neuron;
EXPORT_VAR uint32_t* recordSpkleft_neuron;
EXPORT_VAR uint32_t* d_recordSpkleft_neuron;
EXPORT_VAR scalar* Vleft_neuron;
EXPORT_VAR scalar* d_Vleft_neuron;
EXPORT_VAR scalar* RefracTimeleft_neuron;
EXPORT_VAR scalar* d_RefracTimeleft_neuron;
#define spikeCount_right_neuron glbSpkCntright_neuron[0]
#define spike_right_neuron glbSpkright_neuron
#define glbSpkShiftright_neuron 0

EXPORT_VAR unsigned int* glbSpkCntright_neuron;
EXPORT_VAR unsigned int* d_glbSpkCntright_neuron;
EXPORT_VAR unsigned int* glbSpkright_neuron;
EXPORT_VAR unsigned int* d_glbSpkright_neuron;
EXPORT_VAR uint32_t* recordSpkright_neuron;
EXPORT_VAR uint32_t* d_recordSpkright_neuron;
EXPORT_VAR scalar* Vright_neuron;
EXPORT_VAR scalar* d_Vright_neuron;
EXPORT_VAR scalar* RefracTimeright_neuron;
EXPORT_VAR scalar* d_RefracTimeright_neuron;
#define spikeCount_up_neuron glbSpkCntup_neuron[0]
#define spike_up_neuron glbSpkup_neuron
#define glbSpkShiftup_neuron 0

EXPORT_VAR unsigned int* glbSpkCntup_neuron;
EXPORT_VAR unsigned int* d_glbSpkCntup_neuron;
EXPORT_VAR unsigned int* glbSpkup_neuron;
EXPORT_VAR unsigned int* d_glbSpkup_neuron;
EXPORT_VAR uint32_t* recordSpkup_neuron;
EXPORT_VAR uint32_t* d_recordSpkup_neuron;
EXPORT_VAR scalar* Vup_neuron;
EXPORT_VAR scalar* d_Vup_neuron;
EXPORT_VAR scalar* RefracTimeup_neuron;
EXPORT_VAR scalar* d_RefracTimeup_neuron;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSyninhibitory_down_neuron;
EXPORT_VAR float* d_inSyninhibitory_down_neuron;
EXPORT_VAR float* inSynexcitatory_down_neuron;
EXPORT_VAR float* d_inSynexcitatory_down_neuron;
EXPORT_VAR float* inSyninput_to_high_filter;
EXPORT_VAR float* d_inSyninput_to_high_filter;
EXPORT_VAR float* inSynhigh_to_low;
EXPORT_VAR float* d_inSynhigh_to_low;
EXPORT_VAR float* inSyninput_to_low_filter;
EXPORT_VAR float* d_inSyninput_to_low_filter;
EXPORT_VAR float* inSyninhibitory_left_neuron;
EXPORT_VAR float* d_inSyninhibitory_left_neuron;
EXPORT_VAR float* inSynexcitatory_left_neuron;
EXPORT_VAR float* d_inSynexcitatory_left_neuron;
EXPORT_VAR float* inSyninhibitory_right_neuron;
EXPORT_VAR float* d_inSyninhibitory_right_neuron;
EXPORT_VAR float* inSynexcitatory_right_neuron;
EXPORT_VAR float* d_inSynexcitatory_right_neuron;
EXPORT_VAR float* inSyninhibitory_up_neuron;
EXPORT_VAR float* d_inSyninhibitory_up_neuron;
EXPORT_VAR float* inSynexcitatory_up_neuron;
EXPORT_VAR float* d_inSynexcitatory_up_neuron;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthhigh_to_low;
EXPORT_VAR unsigned int* rowLengthhigh_to_low;
EXPORT_VAR unsigned int* d_rowLengthhigh_to_low;
EXPORT_VAR uint32_t* indhigh_to_low;
EXPORT_VAR uint32_t* d_indhigh_to_low;
EXPORT_VAR const unsigned int maxRowLengthinput_to_high_filter;
EXPORT_VAR unsigned int* rowLengthinput_to_high_filter;
EXPORT_VAR unsigned int* d_rowLengthinput_to_high_filter;
EXPORT_VAR uint32_t* indinput_to_high_filter;
EXPORT_VAR uint32_t* d_indinput_to_high_filter;
EXPORT_VAR const unsigned int maxRowLengthinput_to_low_filter;
EXPORT_VAR unsigned int* rowLengthinput_to_low_filter;
EXPORT_VAR unsigned int* d_rowLengthinput_to_low_filter;
EXPORT_VAR uint32_t* indinput_to_low_filter;
EXPORT_VAR uint32_t* d_indinput_to_low_filter;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gexcitatory_down_neuron;
EXPORT_VAR scalar* d_gexcitatory_down_neuron;
EXPORT_VAR scalar* gexcitatory_left_neuron;
EXPORT_VAR scalar* d_gexcitatory_left_neuron;
EXPORT_VAR scalar* gexcitatory_right_neuron;
EXPORT_VAR scalar* d_gexcitatory_right_neuron;
EXPORT_VAR scalar* gexcitatory_up_neuron;
EXPORT_VAR scalar* d_gexcitatory_up_neuron;
EXPORT_VAR scalar* ghigh_to_low;
EXPORT_VAR scalar* d_ghigh_to_low;
EXPORT_VAR scalar* ginhibitory_down_neuron;
EXPORT_VAR scalar* d_ginhibitory_down_neuron;
EXPORT_VAR scalar* ginhibitory_left_neuron;
EXPORT_VAR scalar* d_ginhibitory_left_neuron;
EXPORT_VAR scalar* ginhibitory_right_neuron;
EXPORT_VAR scalar* d_ginhibitory_right_neuron;
EXPORT_VAR scalar* ginhibitory_up_neuron;
EXPORT_VAR scalar* d_ginhibitory_up_neuron;
EXPORT_VAR scalar* ginput_to_high_filter;
EXPORT_VAR scalar* d_ginput_to_high_filter;
EXPORT_VAR scalar* ginput_to_low_filter;
EXPORT_VAR scalar* d_ginput_to_low_filter;

EXPORT_FUNC void pushdown_neuronSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldown_neuronSpikesFromDevice();
EXPORT_FUNC void pushdown_neuronCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldown_neuronCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getdown_neuronCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getdown_neuronCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVdown_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVdown_neuronFromDevice();
EXPORT_FUNC void pushCurrentVdown_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVdown_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentVdown_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimedown_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimedown_neuronFromDevice();
EXPORT_FUNC void pushCurrentRefracTimedown_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimedown_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimedown_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushdown_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldown_neuronStateFromDevice();
EXPORT_FUNC void pushfilter_high_popSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullfilter_high_popSpikesFromDevice();
EXPORT_FUNC void pushfilter_high_popCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullfilter_high_popCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getfilter_high_popCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getfilter_high_popCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVfilter_high_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVfilter_high_popFromDevice();
EXPORT_FUNC void pushCurrentVfilter_high_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVfilter_high_popFromDevice();
EXPORT_FUNC scalar* getCurrentVfilter_high_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimefilter_high_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimefilter_high_popFromDevice();
EXPORT_FUNC void pushCurrentRefracTimefilter_high_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimefilter_high_popFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimefilter_high_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushfilter_high_popStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullfilter_high_popStateFromDevice();
EXPORT_FUNC void pushfilter_low_popSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullfilter_low_popSpikesFromDevice();
EXPORT_FUNC void pushfilter_low_popCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullfilter_low_popCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getfilter_low_popCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getfilter_low_popCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVfilter_low_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVfilter_low_popFromDevice();
EXPORT_FUNC void pushCurrentVfilter_low_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVfilter_low_popFromDevice();
EXPORT_FUNC scalar* getCurrentVfilter_low_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimefilter_low_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimefilter_low_popFromDevice();
EXPORT_FUNC void pushCurrentRefracTimefilter_low_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimefilter_low_popFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimefilter_low_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushfilter_low_popStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullfilter_low_popStateFromDevice();
EXPORT_FUNC void pushinputSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinputSpikesFromDevice();
EXPORT_FUNC void pushinputCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinputCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getinputCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getinputCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushinputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinputStateFromDevice();
EXPORT_FUNC void allocateinputinput(unsigned int count);
EXPORT_FUNC void freeinputinput();
EXPORT_FUNC void pushinputinputToDevice(unsigned int count);
EXPORT_FUNC void pullinputinputFromDevice(unsigned int count);
EXPORT_FUNC void pushleft_neuronSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullleft_neuronSpikesFromDevice();
EXPORT_FUNC void pushleft_neuronCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullleft_neuronCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getleft_neuronCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getleft_neuronCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVleft_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVleft_neuronFromDevice();
EXPORT_FUNC void pushCurrentVleft_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVleft_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentVleft_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeleft_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeleft_neuronFromDevice();
EXPORT_FUNC void pushCurrentRefracTimeleft_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeleft_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeleft_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushleft_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullleft_neuronStateFromDevice();
EXPORT_FUNC void pushright_neuronSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullright_neuronSpikesFromDevice();
EXPORT_FUNC void pushright_neuronCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullright_neuronCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getright_neuronCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getright_neuronCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVright_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVright_neuronFromDevice();
EXPORT_FUNC void pushCurrentVright_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVright_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentVright_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeright_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeright_neuronFromDevice();
EXPORT_FUNC void pushCurrentRefracTimeright_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeright_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeright_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushright_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullright_neuronStateFromDevice();
EXPORT_FUNC void pushup_neuronSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullup_neuronSpikesFromDevice();
EXPORT_FUNC void pushup_neuronCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullup_neuronCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getup_neuronCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getup_neuronCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVup_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVup_neuronFromDevice();
EXPORT_FUNC void pushCurrentVup_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVup_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentVup_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeup_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeup_neuronFromDevice();
EXPORT_FUNC void pushCurrentRefracTimeup_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeup_neuronFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeup_neuron(unsigned int batch = 0); 
EXPORT_FUNC void pushup_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullup_neuronStateFromDevice();
EXPORT_FUNC void pushhigh_to_lowConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullhigh_to_lowConnectivityFromDevice();
EXPORT_FUNC void pushinput_to_high_filterConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinput_to_high_filterConnectivityFromDevice();
EXPORT_FUNC void pushinput_to_low_filterConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinput_to_low_filterConnectivityFromDevice();
EXPORT_FUNC void pushgexcitatory_down_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgexcitatory_down_neuronFromDevice();
EXPORT_FUNC void pushinSynexcitatory_down_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynexcitatory_down_neuronFromDevice();
EXPORT_FUNC void pushexcitatory_down_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexcitatory_down_neuronStateFromDevice();
EXPORT_FUNC void pushgexcitatory_left_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgexcitatory_left_neuronFromDevice();
EXPORT_FUNC void pushinSynexcitatory_left_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynexcitatory_left_neuronFromDevice();
EXPORT_FUNC void pushexcitatory_left_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexcitatory_left_neuronStateFromDevice();
EXPORT_FUNC void pushgexcitatory_right_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgexcitatory_right_neuronFromDevice();
EXPORT_FUNC void pushinSynexcitatory_right_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynexcitatory_right_neuronFromDevice();
EXPORT_FUNC void pushexcitatory_right_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexcitatory_right_neuronStateFromDevice();
EXPORT_FUNC void pushgexcitatory_up_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgexcitatory_up_neuronFromDevice();
EXPORT_FUNC void pushinSynexcitatory_up_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynexcitatory_up_neuronFromDevice();
EXPORT_FUNC void pushexcitatory_up_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullexcitatory_up_neuronStateFromDevice();
EXPORT_FUNC void pushghigh_to_lowToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullghigh_to_lowFromDevice();
EXPORT_FUNC void pushinSynhigh_to_lowToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynhigh_to_lowFromDevice();
EXPORT_FUNC void pushhigh_to_lowStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullhigh_to_lowStateFromDevice();
EXPORT_FUNC void pushginhibitory_down_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginhibitory_down_neuronFromDevice();
EXPORT_FUNC void pushinSyninhibitory_down_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninhibitory_down_neuronFromDevice();
EXPORT_FUNC void pushinhibitory_down_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinhibitory_down_neuronStateFromDevice();
EXPORT_FUNC void pushginhibitory_left_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginhibitory_left_neuronFromDevice();
EXPORT_FUNC void pushinSyninhibitory_left_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninhibitory_left_neuronFromDevice();
EXPORT_FUNC void pushinhibitory_left_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinhibitory_left_neuronStateFromDevice();
EXPORT_FUNC void pushginhibitory_right_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginhibitory_right_neuronFromDevice();
EXPORT_FUNC void pushinSyninhibitory_right_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninhibitory_right_neuronFromDevice();
EXPORT_FUNC void pushinhibitory_right_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinhibitory_right_neuronStateFromDevice();
EXPORT_FUNC void pushginhibitory_up_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginhibitory_up_neuronFromDevice();
EXPORT_FUNC void pushinSyninhibitory_up_neuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninhibitory_up_neuronFromDevice();
EXPORT_FUNC void pushinhibitory_up_neuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinhibitory_up_neuronStateFromDevice();
EXPORT_FUNC void pushginput_to_high_filterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginput_to_high_filterFromDevice();
EXPORT_FUNC void pushinSyninput_to_high_filterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninput_to_high_filterFromDevice();
EXPORT_FUNC void pushinput_to_high_filterStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinput_to_high_filterStateFromDevice();
EXPORT_FUNC void pushginput_to_low_filterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginput_to_low_filterFromDevice();
EXPORT_FUNC void pushinSyninput_to_low_filterToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninput_to_low_filterFromDevice();
EXPORT_FUNC void pushinput_to_low_filterStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinput_to_low_filterStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);
EXPORT_FUNC void pullRecordingBuffersFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t, unsigned int recordingTimestep); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
