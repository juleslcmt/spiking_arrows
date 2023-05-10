import matplotlib.pyplot as plt
import numpy as np
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Downscaled resolution
width = 8
height = 6

# Resonate and fire neuron model
rf_model = create_custom_neuron_class("RF", param_names=["Damp", "Omega"],
                                      var_name_types=[("V", "scalar"), ("U", "scalar")],
                                      sim_code=
                                      """
                                      const scalar oldV = $(V);
                                      const scalar oldU = $(U);
                                      $(V) += DT * oldU;
                                      $(U) += DT * ($(Isyn) - (2.0 * $(Damp) * oldU) - ($(Omega) * $(Omega) * oldV));
                                      """,
                                      threshold_condition_code="""$(V) >= 1.0""",
                                      reset_code="""""")

model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
model.dT = 0.1

# Neuron parameters
excitatory_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.75,
                 'Ioffset': 0}
inhibitory_params = {"C": 1.0, "TauM": 1.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -47.85,
                 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.75,
                 'Ioffset': 0}
LIF_init = {'RefracTime': 0, 'V': -65}

# Mapping input spikes (test)
spike_times = [np.arange(0, 100, 1000/2000), np.arange(0, 200, 1000/3000), np.arange(100, 200, 1000/2000)] 
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
input_pop.spike_recording_enabled = True

# Network architecture
excitatory_pop = model.add_neuron_population("excitatory_pop", width * height, "LIF", excitatory_params, LIF_init)
excitatory_pop.spike_recording_enabled = True
inhibitory_pop = model.add_neuron_population("inhibitory_pop", width * height, "LIF", inhibitory_params, LIF_init)
inhibitory_pop.spike_recording_enabled = True

up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, LIF_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, LIF_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, LIF_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, LIF_init)
up_neuron.spike_recording_enabled = True
down_neuron.spike_recording_enabled = True
left_neuron.spike_recording_enabled = True
right_neuron.spike_recording_enabled = True

# Input to excitatory and inhibitory matrices
# TODO g value was added randomly
model.add_synapse_population("input_excitatory", "SPARSE_GLOBALG", 0,
                             input_pop, excitatory_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
model.add_synapse_population("input_inhibitory", "SPARSE_GLOBALG", 0,
                             input_pop, inhibitory_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
# Inhibitory matrix inhibits excitatory matrix to filter the source point
model.add_synapse_population("target_filter", "SPARSE_GLOBALG", 0,
                             inhibitory_pop, excitatory_pop,
                             "StaticPulse", {}, {"g": -1000.0}, {}, {},
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
                             excitatory_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             inhibitory_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             excitatory_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             inhibitory_pop, down_neuron,
                             "StaticPulse", {}, {"g": -1 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             excitatory_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             inhibitory_pop, left_neuron,
                             "StaticPulse", {}, {"g": -1 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             excitatory_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             inhibitory_pop, right_neuron,
                             "StaticPulse", {}, {"g": -1 * width_right_weight_matrix}, {}, {},
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

excitatory_spike_times, excitatory_ids = excitatory_pop.spike_recording_data
inhibitory_spike_times, inhibitory_ids = inhibitory_pop.spike_recording_data
up_spike_times, _ = up_neuron.spike_recording_data
down_spike_times, _ = down_neuron.spike_recording_data
left_spike_times, _ = left_neuron.spike_recording_data
right_spike_times, _ = right_neuron.spike_recording_data

timesteps = np.arange(0.0, 200.0, model.dT)

# Create figure with 4 axes
fig, axes = plt.subplots(6,1)
axes[0].scatter(excitatory_spike_times, excitatory_ids,s=4)
axes[0].set_xlabel("time [ms]")
axes[0].set_ylim((0, width*height))
axes[1].scatter(inhibitory_spike_times, inhibitory_ids,s=4)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylim((0, width*height))
axes[2].vlines(up_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[2].set_xlabel("time [ms]")
axes[2].set_xlim((0, 200))
axes[2].set_ylim((0, 4))
axes[3].vlines(down_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[3].set_xlabel("time [ms]")
axes[3].set_xlim((0, 200))
axes[3].set_ylim((0, 4))
axes[4].vlines(left_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[4].set_xlabel("time [ms]")
axes[4].set_xlim((0, 200))
axes[4].set_ylim((0, 4))
axes[5].vlines(right_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[5].set_xlabel("time [ms]")
axes[5].set_xlim((0, 200))
axes[5].set_ylim((0, 4))
plt.show()