import matplotlib.pyplot as plt
import numpy as np
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Simulation variables
check_time = 10  # Check output spikes each 10 ms

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
rf_params = {"Damp": 2.5, "Omega": 1.1 * np.pi * 2.0}
rf_init = {"V": 0.0, "U": 0.0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.75,
                 'Ioffset': 0}
output_init = {'RefracTime': 0, 'V': -65}

# Testing input layer
spike_times_1 = [1.1 * 2 * np.pi * i for i in range(15)]
spike_times_2 = [(0.7 * np.pi * i) for i in range(15)]
# list of sorted spike times for each neurons where t_1_2 is the time of the second spike you want the first neuron to emit
spike_times = [[] for i in range(height * width)]
spiking_neurons = [5, 27]
spike_times[5] = spike_times_1
spike_times[27] = spike_times_2

# count how many spikes each neuron will emit
spikes_per_neuron = [len(n) for n in spike_times]
# calculate cumulative sum i.e. index of the END of each neuron's block of spikes
end_spikes = [int(np.sum(spikes_per_neuron[:i + 1])) if i in spiking_neurons else 0 for i in range(height * width)]
# np.cumsum(spikes_per_neuron)

# from these get index of START of each neurons block of spikes
start_spikes = [0 for _ in range(height * width)]
start_spikes[27] = end_spikes[5]

input_pop = model.add_neuron_population("Input", height * width, "SpikeSourceArray", {},
                                        {"startSpike": start_spikes, "endSpike": end_spikes})
input_pop.set_extra_global_param("spikeTimes", np.concatenate(spike_times))
input_pop.spike_recording_enabled = True

# Network architecture
excitatory_pop = model.add_neuron_population("excitatory_pop", width * height, rf_model, rf_params, rf_init)
inhibitory_pop = model.add_neuron_population("inhibitory_pop", width * height, rf_model, rf_params, rf_init)
up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, output_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, output_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, output_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, output_init)
up_neuron.spike_recording_enabled = True

# input to RF neurons
# TODO g value was added randomly
model.add_synapse_population("InputNeuron", "SPARSE_GLOBALG", 0,
                             input_pop, excitatory_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))

# - Weight matrices -
height_up_weight_vector = np.linspace(1, 0, height)
height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width,))
height_down_weight_vector = np.linspace(0, 1, height)
height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width, 1))
width_right_weight_vector = np.linspace(0, 1, width)
width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width, 1))
width_left_weight_vector = np.linspace(1, 0, width)
width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width, 1))

model.add_synapse_population("excitatory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             excitatory_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

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

up_spike_times, _ = up_neuron.spike_recording_data

timesteps = np.arange(0.0, 200.0, 0.1)

# Create figure with 4 axes
fig, axis = plt.subplots()
axis.plot(timesteps, v)
axis.vlines(up_spike_times, ymin=0, ymax=2, color="red", linestyle="--", label="UP")
axis.set_xlabel("time [ms]")
axis.set_ylabel("V [mV]")
axis.legend()

axis.set_ylim((0, 2))
axis.set_ylim((0, 2))

plt.show()
