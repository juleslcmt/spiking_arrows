import matplotlib.pyplot as plt
import numpy as np
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Downscaled resolution
width = 8
height = 6

# Resonate and fire neuron model
rf_model = create_custom_neuron_class("RF",
                                      param_names=["Damp", "Omega"],
                                      var_name_types=[("V", "scalar"), ("U", "scalar")],
                                      sim_code=
                                      """
                                      scalar oldV = $(V);
                                      scalar oldU = $(U);
                                      $(V) += DT * oldU;
                                      $(U) += DT * ($(Isyn) - (2.0 * $(Damp) * oldU) - ($(Omega) * $(Omega) * oldV));
                                      """,
                                      threshold_condition_code=
                                      """
                                      $(V) >= 1.0
                                      """,
                                      reset_code=
                                      """
                                      """)

model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
model.dT = 0.1
source_freq = 200
target_freq = 300

# Neuron parameters
rf_excitatory_params = {"Damp": 10, "Omega": (source_freq / 100) * np.pi * 2.0}
rf_inhibitory_params = {"Damp": 40, "Omega": (source_freq / 100) * np.pi * 2.0}
rf_init = {"V": 0.0, "U": 0.0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.75,
                 'Ioffset': 0}
output_init = {'RefracTime': 0, 'V': -65}


# Mapping input spikes (test)
def set_input_frequency(spiking_neurons, spike_times, num_neurons):
    start_spikes = np.zeros(num_neurons)
    end_spikes = np.zeros(num_neurons)

    spikes = []
    e = 0
    t = 0
    for i in range(num_neurons):
        if i in spiking_neurons:
            spikes.extend(spike_times[t])
            start_spikes[i] = e
            e += len(spike_times[t])
            end_spikes[i] = e
            t += 1
        else:
            start_spikes[i] = e
            end_spikes[i] = e

    return spikes, start_spikes, end_spikes


spike_times, start_spikes, end_spikes = set_input_frequency([5, 27],
                                                            [[(source_freq / 100) * 2 * np.pi * i for i in range(15)],
                                                             [(target_freq / 100) * 2 * np.pi * i for i in range(15)]],
                                                            height * width)

input_pop = model.add_neuron_population("input_pop", height * width, "SpikeSourceArray", {},
                                        {"startSpike": start_spikes, "endSpike": end_spikes})
input_pop.set_extra_global_param("spikeTimes", spike_times)
input_pop.spike_recording_enabled = True

# Network architecture
excitatory_pop = model.add_neuron_population("excitatory_pop", width * height, rf_model, rf_excitatory_params, rf_init)
excitatory_pop.spike_recording_enabled = True
inhibitory_pop = model.add_neuron_population("inhibitory_pop", width * height, rf_model, rf_inhibitory_params, rf_init)
inhibitory_pop.spike_recording_enabled = True

up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, output_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, output_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, output_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, output_init)
up_neuron.spike_recording_enabled = True

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

# Weight matrices
height_up_weight_vector = np.linspace(1, 0, height)
height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width,))
height_down_weight_vector = np.linspace(0, 1, height)
height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width, 1))
width_right_weight_vector = np.linspace(0, 1, width)
width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width, 1))
width_left_weight_vector = np.linspace(1, 0, width)
width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width, 1))

# Excitatory and inhibitory matrices to directional neurons
model.add_synapse_population("excitatory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             excitatory_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             inhibitory_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=2000)

v_exc_view = excitatory_pop.vars["V"].view
v_inh_view = inhibitory_pop.vars["V"].view
v_exc = []
v_inh = []
while model.t < 200.0:
    model.step_time()
    excitatory_pop.pull_var_from_device("V")
    inhibitory_pop.pull_var_from_device("V")
    v_exc.append(np.copy(v_exc_view))
    v_inh.append(np.copy(v_inh_view))

model.pull_recording_buffers_from_device()

input_spike_times, input_spike_ids = input_pop.spike_recording_data
excitatory_spike_times, excitatory_spike_ids = excitatory_pop.spike_recording_data
inhibitory_spike_times, inhibitory_spike_ids = inhibitory_pop.spike_recording_data
up_spike_times, _ = up_neuron.spike_recording_data

timesteps = np.arange(0.0, 200.0, model.dT)

fig, axes = plt.subplots(4, 1)
axes[0].scatter(input_spike_times, input_spike_ids, s=1)
axes[0].set_xlabel("time [ms]")
axes[0].set_ylim((0, width * height))
axes[0].set_title('Input population')
axes[1].scatter(excitatory_spike_times, excitatory_spike_ids, s=1)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylim((0, width * height))
axes[1].set_title('Excitatory population')
axes[2].scatter(inhibitory_spike_times, inhibitory_spike_ids, s=1)
axes[2].set_xlabel("time [ms]")
axes[2].set_ylim((0, width * height))
axes[2].set_title('Inhibitory population')
axes[3].vlines(up_spike_times, ymin=0, ymax=1, color="red", linestyle="--")
axes[3].set_xlabel("time [ms]")
axes[3].set_ylim((0, 4))
axes[3].set_title('Output population')
fig.tight_layout()
fig.show()

# Plot membrane potential
fig, axes = plt.subplots(2, 1)
axes[0].plot(timesteps, np.array(v_exc)[:, 5], label="neuron 5")
axes[0].plot(timesteps, np.array(v_exc)[:, 27], label="neuron 27")
axes[0].set_xlabel("time [ms]")
axes[0].set_ylabel("membrane potential [V]")
axes[0].set_title("Excitatory neurons")
axes[1].plot(timesteps, np.array(v_inh)[:, 5])
axes[1].plot(timesteps, np.array(v_inh)[:, 27])
axes[1].set_xlabel("time [ms]")
axes[1].set_ylabel("membrane potential [V]")
axes[1].set_title("Inhibitory neurons")
fig.tight_layout()
fig.legend()
fig.show()
