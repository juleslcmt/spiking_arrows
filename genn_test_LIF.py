import matplotlib.pyplot as plt
import numpy as np
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Resolution
width = 640
height = 480

# Resonate and fire neuron model
model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
model.dT = 0.1

# Neuron parameters
filter_high_params = {"C": 1.0, "TauM": 0.1, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
filter_low_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.95,
                 'Ioffset': 0}
LIF_init = {'RefracTime': 0, 'V': -65}

# Mapping input spikes (test)
spike_times = [np.arange(0, 100, 1000/5000), np.arange(0, 200, 1000/10000), np.arange(100, 200, 1000/5000)] 
len_spike_times = [len(x) for x in spike_times]

start_spikes = [0 for i in range(height*width)]
end_spikes = [0 for i in range(height*width)]

end_spikes[0] = len_spike_times[0]
start_spikes[width-1] = len_spike_times[0]
end_spikes[width-1] = len_spike_times[0] + len_spike_times[1]
start_spikes[width*int(height/4)-1] = len_spike_times[0] + len_spike_times[1]
end_spikes[width*int(height/4)-1] = len_spike_times[0] + len_spike_times[1] + len_spike_times[2]

input_pop = model.add_neuron_population("input_pop", height * width, "SpikeSourceArray", {},
                                        {"startSpike": start_spikes, "endSpike": end_spikes})
input_pop.set_extra_global_param("spikeTimes", np.concatenate(spike_times, axis=None))
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
max_weight_v = height*10
min_weight_v = height
max_weight_h = width*10
min_weight_h = width

height_up_weight_vector = np.linspace(max_weight_v, min_weight_v, height)
height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width,))

height_down_weight_vector = np.linspace(min_weight_v, max_weight_v, height)
height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width,))

width_left_weight_vector = np.linspace(max_weight_h, min_weight_h, width)
width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width,))

width_right_weight_vector = np.linspace(min_weight_h, max_weight_h, width)
width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width,))

# Filter high to directional neurons
model.add_synapse_population("excitatory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
# Filter low to directional neurons
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1.25 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, down_neuron,
                             "StaticPulse", {}, {"g": -1.25 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, left_neuron,
                             "StaticPulse", {}, {"g": -1.25 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, right_neuron,
                             "StaticPulse", {}, {"g": -1.25 * width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=2000)

v_view = up_neuron.vars["V"].view
v = []
while model.t < 200.0:
    model.step_time()
    up_neuron.pull_var_from_device("V")
    v.append(v_view[0])

    # if model.t % check_time == 0:
    #     model.pull_recording_buffers_from_device()
    #     directional_spikes, _ = up_neuron.spike_recording_data
    #     print(directional_spikes)

model.pull_recording_buffers_from_device()

filter_high_spike_times, excitatory_ids = filter_high_pop.spike_recording_data
filter_low_spike_times, inhibitory_ids = filter_low_pop.spike_recording_data
up_spike_times, _ = up_neuron.spike_recording_data
down_spike_times, _ = down_neuron.spike_recording_data
left_spike_times, _ = left_neuron.spike_recording_data
right_spike_times, _ = right_neuron.spike_recording_data

timesteps = np.arange(0.0, 200.0, model.dT)

# Create figure with 4 axes
fig, axes = plt.subplots(6,1)
axes[0].scatter(filter_high_spike_times, excitatory_ids,s=4)
axes[0].set_xlabel("time [ms]")
axes[0].set_xlim((0, 200))
axes[0].set_ylim((0, width*height))
axes[0].set_title("Filter High")
axes[1].scatter(filter_low_spike_times, inhibitory_ids,s=4)
axes[1].set_xlabel("time [ms]")
axes[1].set_xlim((0, 200))
axes[1].set_ylim((0, width*height))
axes[1].set_title("Filter Low")
axes[2].vlines(up_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[2].set_xlabel("time [ms]")
axes[2].set_xlim((0, 200))
axes[2].set_title("Up neuron")
axes[3].vlines(down_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[3].set_xlabel("time [ms]")
axes[3].set_xlim((0, 200))
axes[3].set_title("Down neuron")
axes[4].vlines(left_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[4].set_xlabel("time [ms]")
axes[4].set_xlim((0, 200))
axes[4].set_title("Left neuron")
axes[5].vlines(right_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[5].set_xlabel("time [ms]")
axes[5].set_xlim((0, 200))
axes[5].set_title("Right neuron")
plt.tight_layout()
plt.show()