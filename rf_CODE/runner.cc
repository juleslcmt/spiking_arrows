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
unsigned int* glbSpkCntInput;
unsigned int* glbSpkInput;
uint32_t* recordSpkInput;
unsigned int* startSpikeInput;
unsigned int* endSpikeInput;
scalar* spikeTimesInput;
unsigned int* glbSpkCntNeuron;
unsigned int* glbSpkNeuron;
uint32_t* recordSpkNeuron;
scalar* VNeuron;
scalar* UNeuron;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynInputNeuron;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocatespikeTimesInput(unsigned int count) {
    spikeTimesInput = new scalar[count];
    pushMergedNeuronUpdate1spikeTimesToDevice(0, spikeTimesInput);
}
void freespikeTimesInput() {
    delete[] spikeTimesInput;
}
void pushspikeTimesInputToDevice(unsigned int count) {
}
void pullspikeTimesInputFromDevice(unsigned int count) {
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushInputSpikesToDevice(bool uninitialisedOnly) {
}

void pushInputCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushstartSpikeInputToDevice(bool uninitialisedOnly) {
}

void pushCurrentstartSpikeInputToDevice(bool uninitialisedOnly) {
}

void pushendSpikeInputToDevice(bool uninitialisedOnly) {
}

void pushCurrentendSpikeInputToDevice(bool uninitialisedOnly) {
}

void pushInputStateToDevice(bool uninitialisedOnly) {
    pushstartSpikeInputToDevice(uninitialisedOnly);
    pushendSpikeInputToDevice(uninitialisedOnly);
}

void pushNeuronSpikesToDevice(bool uninitialisedOnly) {
}

void pushNeuronCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVNeuronToDevice(bool uninitialisedOnly) {
}

void pushCurrentVNeuronToDevice(bool uninitialisedOnly) {
}

void pushUNeuronToDevice(bool uninitialisedOnly) {
}

void pushCurrentUNeuronToDevice(bool uninitialisedOnly) {
}

void pushNeuronStateToDevice(bool uninitialisedOnly) {
    pushVNeuronToDevice(uninitialisedOnly);
    pushUNeuronToDevice(uninitialisedOnly);
}

void pushinSynInputNeuronToDevice(bool uninitialisedOnly) {
}

void pushInputNeuronStateToDevice(bool uninitialisedOnly) {
    pushinSynInputNeuronToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullInputSpikesFromDevice() {
}

void pullInputCurrentSpikesFromDevice() {
}

void pullstartSpikeInputFromDevice() {
}

void pullCurrentstartSpikeInputFromDevice() {
}

void pullendSpikeInputFromDevice() {
}

void pullCurrentendSpikeInputFromDevice() {
}

void pullInputStateFromDevice() {
    pullstartSpikeInputFromDevice();
    pullendSpikeInputFromDevice();
}

void pullNeuronSpikesFromDevice() {
}

void pullNeuronCurrentSpikesFromDevice() {
}

void pullVNeuronFromDevice() {
}

void pullCurrentVNeuronFromDevice() {
}

void pullUNeuronFromDevice() {
}

void pullCurrentUNeuronFromDevice() {
}

void pullNeuronStateFromDevice() {
    pullVNeuronFromDevice();
    pullUNeuronFromDevice();
}

void pullinSynInputNeuronFromDevice() {
}

void pullInputNeuronStateFromDevice() {
    pullinSynInputNeuronFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getInputCurrentSpikes(unsigned int batch) {
    return (glbSpkInput);
}

unsigned int& getInputCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntInput[0];
}

unsigned int* getCurrentstartSpikeInput(unsigned int batch) {
    return startSpikeInput;
}

unsigned int* getCurrentendSpikeInput(unsigned int batch) {
    return endSpikeInput;
}

unsigned int* getNeuronCurrentSpikes(unsigned int batch) {
    return (glbSpkNeuron);
}

unsigned int& getNeuronCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntNeuron[0];
}

scalar* getCurrentVNeuron(unsigned int batch) {
    return VNeuron;
}

scalar* getCurrentUNeuron(unsigned int batch) {
    return UNeuron;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushInputStateToDevice(uninitialisedOnly);
    pushNeuronStateToDevice(uninitialisedOnly);
    pushInputNeuronStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullInputStateFromDevice();
    pullNeuronStateFromDevice();
    pullInputNeuronStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullInputCurrentSpikesFromDevice();
    pullNeuronCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkInput = new uint32_t[numWords];
            pushMergedNeuronUpdate1recordSpkToDevice(0, recordSpkInput);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkNeuron = new uint32_t[numWords];
            pushMergedNeuronUpdate0recordSpkToDevice(0, recordSpkNeuron);
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
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
        }
    }
}

void allocateMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntInput = new unsigned int[1];
    glbSpkInput = new unsigned int[1];
    startSpikeInput = new unsigned int[1];
    endSpikeInput = new unsigned int[1];
    glbSpkCntNeuron = new unsigned int[1];
    glbSpkNeuron = new unsigned int[1];
    VNeuron = new scalar[1];
    UNeuron = new scalar[1];
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    inSynInputNeuron = new float[1];
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    pushMergedNeuronInitGroup0ToDevice(0, glbSpkCntNeuron, glbSpkNeuron, VNeuron, UNeuron, inSynInputNeuron, 1);
    pushMergedNeuronInitGroup1ToDevice(0, glbSpkCntInput, glbSpkInput, startSpikeInput, endSpikeInput, 1);
    pushMergedNeuronUpdateGroup0ToDevice(0, glbSpkCntNeuron, glbSpkNeuron, VNeuron, UNeuron, inSynInputNeuron, recordSpkNeuron, 1);
    pushMergedNeuronUpdateGroup1ToDevice(0, glbSpkCntInput, glbSpkInput, startSpikeInput, endSpikeInput, spikeTimesInput, recordSpkInput, 1);
    pushMergedPresynapticUpdateGroup0ToDevice(0, inSynInputNeuron, glbSpkCntInput, glbSpkInput, 1, 1, 1);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, glbSpkCntNeuron);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(0, glbSpkCntInput);
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
    delete[] glbSpkCntInput;
    delete[] glbSpkInput;
    delete[] recordSpkInput;
    delete[] startSpikeInput;
    delete[] endSpikeInput;
    delete[] glbSpkCntNeuron;
    delete[] glbSpkNeuron;
    delete[] recordSpkNeuron;
    delete[] VNeuron;
    delete[] UNeuron;
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    delete[] inSynInputNeuron;
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

size_t getFreeDeviceMemBytes() {
    return 0;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

