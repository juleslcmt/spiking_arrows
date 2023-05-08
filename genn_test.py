import numpy as np
import matplotlib.pyplot as plt

from pygenn.genn_model import GeNNModel, create_custom_neuron_class, create_custom_current_source_class

# Simulation variables
model = None
excitatory_matrix = None
inhibitory_matrix = None
directional_neurons = None
check_time = 10  # Check output spikes each 10 ms

# Downscaled resolution
width = 8
height = 6


def build_network():
    global model, excitatory_matrix, inhibitory_matrix, directional_neurons

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

    # Network architecture
    excitatory_matrix = model.add_neuron_population("excitatory_pop", width * height, rf_model, rf_params, rf_init)
    inhibitory_matrix = model.add_neuron_population("inhibitory_pop", width * height, rf_model, rf_params, rf_init)
    directional_neurons = model.add_neuron_population("directional_pop", 4, "LIF")
    directional_neurons.spike_recording_enabled = True

    # TODO: Selecting neurons in the population? Fixing static weights?
    model.add_synapse_population("excitatory_up_neuron", "DENSE_GLOBALG", 0,
                                 excitatory_matrix, directional_neurons[0],
                                 "StaticPulse", {}, {"g": 70.0}, {}, {},
                                 "DeltaCurr", {}, {})

    model.build()
    model.load()


def simulate():
    while model.t < 200.0:
        model.step_time()

        if model.t % check_time == 0:
            model.pull_recording_buffers_from_device()
            directional_spikes, _ = directional_neurons.spike_recording_data
            print(directional_spikes)


if __name__ == '__main__':
    build_network()
    simulate()




