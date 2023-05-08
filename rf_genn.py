import numpy as np
import matplotlib.pyplot as plt

from pygenn.genn_model import GeNNModel, create_custom_neuron_class, create_custom_current_source_class

# Resonate and fire neuron model
rf_matlab_model = create_custom_neuron_class(
    "RF",
    param_names=["Damp", "Omega"],
    var_name_types=[("V", "scalar"), ("U", "scalar")],
    sim_code=
    """
    const scalar oldV = $(V);
    const scalar oldU = $(U);
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

rf_params = {"Damp": 2.5, "Omega": 1.1 * np.pi * 2.0}
rf_init = {"V": 0.0, "U": 0.0}

neuron_pop = model.add_neuron_population("Neuron", 1, rf_matlab_model, rf_params, rf_init)
neuron_pop.spike_recording_enabled = True

spike_times = [1.1 * 2 * np.pi * i for i in range(15)] 
spike_times.extend([150.0 + (0.7 * np.pi * i) for i in range(15)])
input_pop = model.add_neuron_population("Input", 1, "SpikeSourceArray", {}, {"startSpike": 0, "endSpike": len(spike_times)});
input_pop.set_extra_global_param("spikeTimes", spike_times)
input_pop.spike_recording_enabled = True

model.add_synapse_population("InputNeuron", "DENSE_GLOBALG", 0, 
                             input_pop, neuron_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {}, 
                             "DeltaCurr", {}, {})
                             
model.build()
model.load(num_recording_timesteps=2000)

v_view = neuron_pop.vars["V"].view

v = []
while model.t < 200.0:
    model.step_time()
    neuron_pop.pull_var_from_device("V")
    v.append(v_view[0])

model.pull_recording_buffers_from_device()

input_spike_times, _ = input_pop.spike_recording_data
neuron_spike_times, _ = neuron_pop.spike_recording_data

timesteps = np.arange(0.0, 200.0, 0.1)

# Create figure with 4 axes
fig, axis = plt.subplots()
axis.plot(timesteps, v)
axis.vlines(input_spike_times, ymin=0, ymax=2, color="red", linestyle="--", label="input")
axis.vlines(neuron_spike_times, ymin=0, ymax=2, color="blue", label="neuron")
axis.set_xlabel("time [ms]")
axis.set_ylabel("V [mV]")
axis.legend()

axis.set_ylim((0, 2))
axis.set_ylim((0, 2))

plt.show()