import matplotlib.pyplot as plt
from pygenn.genn_model import GeNNModel, init_connectivity

from helper_functions import *

# Resolution
width = 640
height = 480

# Resonate and fire neuron model
model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
model.dT = 0.1
sim_time = 200
timesteps = int(sim_time / model.dT)

# Neuron parameters
filter_source_params = {"C": 1.0, "TauM": 0.1, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5,
                        'Ioffset': 0}
filter_target_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5,
                        'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 0.5, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -54.5,
                 'Ioffset': 0}
LIF_init = {'RefracTime': 0, 'V': -65}

RAF_source_params = {"Damp": 0.1, "Omega": 100 / 1000 * np.pi * 2}
RAF_target_params = {"Damp": 0.1, "Omega": 300 / 1000 * np.pi * 2}
RAF_init = {"V": 0.0, "U": 0.0}

# Mapping input spikes (test)
frequencies = [100, 300]
source_neuron_ij = (10, 10)
target_neuron_ij = (20, 10)
source_neuron = np.ravel_multi_index(source_neuron_ij, (width, height))
target_neuron = np.ravel_multi_index(target_neuron_ij, (width, height))
spike_times = []
for f in frequencies:
    spike_times.append(generate_spike_times_frequency(0, 100, f))
spike_times, start_spikes, end_spikes = set_input_frequency([source_neuron, target_neuron],
                                                            spike_times, width * height)
input_pop = model.add_neuron_population("input_pop", height * width, "SpikeSourceArray", {},
                                        {"startSpike": start_spikes, "endSpike": end_spikes})
input_pop.set_extra_global_param("spikeTimes", spike_times)
input_pop.spike_recording_enabled = True
input_matrix = np.zeros((height, width))
input_matrix[source_neuron_ij] = frequencies[0]
input_matrix[target_neuron_ij] = frequencies[1]

# Network architecture
filter_target_pop = model.add_neuron_population("filter_high_pop",
                                                width * height,
                                                "LIF",
                                                filter_source_params, LIF_init)
filter_target_pop.spike_recording_enabled = True
filter_source_pop = model.add_neuron_population("filter_low_pop",
                                                width * height,
                                                "LIF",
                                                filter_target_params, LIF_init)
filter_source_pop.spike_recording_enabled = True

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
                             input_pop, filter_target_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))

model.add_synapse_population("input_to_low_filter", "SPARSE_GLOBALG", 0,
                             input_pop, filter_source_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))

model.add_synapse_population("high_to_low", "SPARSE_GLOBALG", 0,
                             filter_target_pop, filter_source_pop,
                             "StaticPulse", {}, {"g": -1400.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))

# Weight matrices
max_weight_v = height * 10
min_weight_v = height
max_weight_h = width * 10
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
                             filter_target_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_target_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_target_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_target_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Filter low to directional neurons
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_source_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1.25 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_source_pop, down_neuron,
                             "StaticPulse", {}, {"g": -1.25 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_source_pop, left_neuron,
                             "StaticPulse", {}, {"g": -1.25 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_source_pop, right_neuron,
                             "StaticPulse", {}, {"g": -1.25 * width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=timesteps)

v_view = up_neuron.vars["V"].view
v = []
while model.t < sim_time:
    model.step_time()
    up_neuron.pull_var_from_device("V")
    v.append(v_view[0])

model.pull_recording_buffers_from_device()

filter_target_spike_times, target_spike_ids = filter_target_pop.spike_recording_data
filter_source_spike_times, source_spike_ids = filter_source_pop.spike_recording_data
up_spike_times, _ = up_neuron.spike_recording_data
down_spike_times, _ = down_neuron.spike_recording_data
left_spike_times, _ = left_neuron.spike_recording_data
right_spike_times, _ = right_neuron.spike_recording_data

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
t = np.arange(0.0, sim_time, model.dT)
cax = axes[0].matshow(input_matrix)
axes[0].legend(['Source', 'Target'])
fig.colorbar(cax)
axes[1].scatter(filter_target_spike_times, target_spike_ids, color='red', s=4, label='Excitatory')
axes[1].scatter(filter_source_spike_times, source_spike_ids, color='blue', s=4, label='Inhibitory')
axes[1].set_ylim((0, width * height))
axes[1].set_title("Input layers", fontsize=20)
axes[1].legend()
axes[2].vlines(up_spike_times, ymin=3, ymax=4, color='green', label='Up')
axes[2].vlines(down_spike_times, ymin=2, ymax=3, color='red', label='Down')
axes[2].vlines(left_spike_times, ymin=1, ymax=2, color='blue', label='Left')
axes[2].vlines(right_spike_times, ymin=0, ymax=1, color='orange', label='Right')
axes[2].set_ylim((0, 4))
axes[2].set_xlabel("time [ms]")
axes[2].set_title("Directional neurons", fontsize=20)
axes[2].legend()
plt.tight_layout()
plt.show()
