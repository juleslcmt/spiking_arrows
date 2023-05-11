import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/opt/spiking_arrows/')

import laser
from aestream import USBInput
import time
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

check_time = 0.001 

# Downscaled resolution
width = 640
height = 480

genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],
    
    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False)

# Resonate and fire neuron model
model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
model.dT = check_time

# Neuron parameters
filter_high_params = {"C": 1.0, "TauM": 1.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -48.0, 'Ioffset': 0}
filter_low_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.95,
                 'Ioffset': 0}
LIF_init = {'RefracTime': 0, 'V': -65}

# Mapping input spikes (test)
""" spike_times = [np.arange(0, 100, 1000/2000), np.arange(0, 200, 1000/3000), np.arange(100, 200, 1000/2000)] 
len_spike_times = [len(x) for x in spike_times]

start_spikes = [0 for i in range(height*width)]
end_spikes = [0 for i in range(height*width)]

end_spikes[0] = len_spike_times[0]
start_spikes[3] = len_spike_times[0]
end_spikes[3] = len_spike_times[0] + len_spike_times[1]
start_spikes[width-1] = len_spike_times[0] + len_spike_times[1]
end_spikes[width-1] = len_spike_times[0] + len_spike_times[1] + len_spike_times[2] 

input_pop = model.add_neuron_population("input_pop", height * width, "SpikeSourceArray", {},
                                        {"startSpike": start_spikes, "endSpike": end_spikes})
input_pop.set_extra_global_param("spikeTimes", np.concatenate(spike_times, axis=None))
input_pop.spike_recording_enabled = True"""

model = GeNNModel("float", "usb_genn")
input_pop = model.add_neuron_population("input",  height * width, genn_input_model, {}, {})
input_pop.set_extra_global_param("input", np.empty(307200, dtype=np.uint32))
input_pop.spike_recording_enabled = True

# Network architecture
filter_high_pop = model.add_neuron_population("filter_high_pop", width * height, "LIF", filter_high_params, LIF_init)
filter_high_pop.spike_recording_enabled = True
filter_low_pop = model.add_neuron_population("filter_low_pop", width * height, "LIF", filter_low_params, LIF_init)
filter_low_pop.spike_recording_enabled = True

up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, LIF_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, LIF_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, LIF_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, LIF_init)
up_neuron.spike_recording_enabled = True
down_neuron.spike_recording_enabled = True
left_neuron.spike_recording_enabled = True
right_neuron.spike_recording_enabled = True

# Input to filter matrices
model.add_synapse_population("input_to_high_filter", "SPARSE_GLOBALG", 0,
                             input_pop, filter_high_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
model.add_synapse_population("input_to_low_filter", "SPARSE_GLOBALG", 0,
                             input_pop, filter_low_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
model.add_synapse_population("high_to_low", "SPARSE_GLOBALG", 0,
                             filter_high_pop, filter_low_pop,
                             "StaticPulse", {}, {"g": -1400.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))

# Weight matrices
height_up_weight_vector = np.linspace(1, 0, height)
height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width,))
height_down_weight_vector = np.linspace(0, 1, height)
height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width,))
width_right_weight_vector = np.linspace(0, 1, width)
width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width,))
width_left_weight_vector = np.linspace(1, 0, width)
width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width,))

# Excitatory and inhibitory matrices to directional neurons
model.add_synapse_population("excitatory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, down_neuron,
                             "StaticPulse", {}, {"g": -1 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, left_neuron,
                             "StaticPulse", {}, {"g": -1 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, right_neuron,
                             "StaticPulse", {}, {"g": -1 * width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=2000)

v_view_up = up_neuron.vars["V"].view
v_view_down = down_neuron.vars["V"].view
v_view_left = left_neuron.vars["V"].view
v_view_right = right_neuron.vars["V"].view

#spikes_history = input_pop.vars["current_spikes"].view

v_all = [v_view_up, v_view_down, v_view_left, v_view_right]

step_size = 100 
arrow_neurons = ["up_neuron","down_neuron","left_neuron","right_neuron"]
moves = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]

with USBInput((height, width), device="genn") as stream:
    with laser.Laser() as l :
        l.on()
        state = (2000, 2000)

        time_start = time.time()
        # Loop forever
        while True:
            #print(stream)
            for i in range(1000000):
                stream.read_genn(input_pop.extra_global_params["input"].view)
                #print("1ms time window with ", np.count_nonzero(input_pop.extra_global_params["input"].view), " events")
                #input_pop.push_extra_global_param_to_device("input")
            
                model.step_time()
                time.sleep(check_time)
                moving = False
                #print(model.pull_prev_spikes_from_device("input"))
                model.pull_recording_buffers_from_device()
                #print(model.pull_current_spikes_from_device("input"))

                print(input_pop.current_spike_events)
                print(filter_low_pop.current_spike_events)
                print(filter_high_pop.current_spike_events)
                #print(model.pull_current_spike_events_from_device("input"))
                #print(model.pull_spike_events_from_device("input"))
                #print(model.pull_spikes_times_from_device("input"))
                for j,neuron in enumerate(arrow_neurons) :
                    if model.pull_spikes_from_device(neuron) :
                        print("spiking " + neuron)
                        state = (max(0,min(4095,state[0]+moves[j][0])), max(0,min(4095,state[1]+moves[j][1])))
                        moving = True
                
                if moving :
                    print(state)
                    l.move(*state)


