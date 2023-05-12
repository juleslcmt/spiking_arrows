#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#define spikeCount_Input glbSpkCntInput[0]
#define spike_Input glbSpkInput
#define glbSpkShiftInput 0

EXPORT_VAR unsigned int* glbSpkCntInput;
EXPORT_VAR unsigned int* glbSpkInput;
EXPORT_VAR uint32_t* recordSpkInput;
EXPORT_VAR unsigned int* startSpikeInput;
EXPORT_VAR unsigned int* endSpikeInput;
EXPORT_VAR scalar* spikeTimesInput;
#define spikeCount_Neuron glbSpkCntNeuron[0]
#define spike_Neuron glbSpkNeuron
#define glbSpkShiftNeuron 0

EXPORT_VAR unsigned int* glbSpkCntNeuron;
EXPORT_VAR unsigned int* glbSpkNeuron;
EXPORT_VAR uint32_t* recordSpkNeuron;
EXPORT_VAR scalar* VNeuron;
EXPORT_VAR scalar* UNeuron;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynInputNeuron;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

EXPORT_FUNC void pushInputSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputSpikesFromDevice();
EXPORT_FUNC void pushInputCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getInputCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getInputCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushstartSpikeInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullstartSpikeInputFromDevice();
EXPORT_FUNC void pushCurrentstartSpikeInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentstartSpikeInputFromDevice();
EXPORT_FUNC unsigned int* getCurrentstartSpikeInput(unsigned int batch = 0); 
EXPORT_FUNC void pushendSpikeInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullendSpikeInputFromDevice();
EXPORT_FUNC void pushCurrentendSpikeInputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentendSpikeInputFromDevice();
EXPORT_FUNC unsigned int* getCurrentendSpikeInput(unsigned int batch = 0); 
EXPORT_FUNC void pushInputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputStateFromDevice();
EXPORT_FUNC void allocatespikeTimesInput(unsigned int count);
EXPORT_FUNC void freespikeTimesInput();
EXPORT_FUNC void pushspikeTimesInputToDevice(unsigned int count);
EXPORT_FUNC void pullspikeTimesInputFromDevice(unsigned int count);
EXPORT_FUNC void pushNeuronSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullNeuronSpikesFromDevice();
EXPORT_FUNC void pushNeuronCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullNeuronCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getNeuronCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getNeuronCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVNeuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVNeuronFromDevice();
EXPORT_FUNC void pushCurrentVNeuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVNeuronFromDevice();
EXPORT_FUNC scalar* getCurrentVNeuron(unsigned int batch = 0); 
EXPORT_FUNC void pushUNeuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullUNeuronFromDevice();
EXPORT_FUNC void pushCurrentUNeuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentUNeuronFromDevice();
EXPORT_FUNC scalar* getCurrentUNeuron(unsigned int batch = 0); 
EXPORT_FUNC void pushNeuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullNeuronStateFromDevice();
EXPORT_FUNC void pushinSynInputNeuronToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynInputNeuronFromDevice();
EXPORT_FUNC void pushInputNeuronStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullInputNeuronStateFromDevice();
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
