import matplotlib.pyplot as plt
import numpy as np
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Resonate and fire neuron model
rf_matlab_model = create_custom_neuron_class(
    "RF",
    param_names=["Damp", "Omega"],
    var_name_types=[("V", "scalar"), ("U", "scalar")],
    support_code=
    """
    SUPPORT_CODE_FUNC scalar dudt( scalar U, scalar Damp, scalar Omega, scalar Isyn, scalar V ){
    return (Isyn - (2.0 * Damp * U) - (Omega * Omega * V) );
    }
    """,
    sim_code=
    """
    const scalar oldV = $(V);
    
    $(V) +=  DT * $(U);
    
    const scalar k1 = dudt( $(U), $(Damp), $(Omega), $(Isyn), oldV );
    const scalar k2 = dudt( $(U) + DT * k1 / 2, $(Damp), $(Omega), $(Isyn), oldV );
    const scalar k3 = dudt( $(U) + DT * k2 / 2, $(Damp), $(Omega), $(Isyn), oldV );
    const scalar k4 = dudt( $(U) + DT * k3, $(Damp), $(Omega), $(Isyn), oldV );
    $(U) += DT * ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6;
    
    if ( $(V) > 1.0 ) {
    $(V) = 1.0;
    }
    """,
    threshold_condition_code=
    """
    $(V) >= 0.99
    """,
    reset_code=
    """
    """)


def generate_spike_times_frequency(start_time, end_time, frequency):
    end_time /= 1000

    sample_rate = 100000
    amplitude = 1
    theta = 0

    time = np.arange(start_time, end_time, 1 / sample_rate)
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    # local min
    maxima = (np.diff(np.sign(np.diff(sinewave))) < 0).nonzero()[0] + 1  # local max

    return time[maxima] * 1000


def set_input_frequency(spiking_neurons, spike_times):
    num_neurons = len(spiking_neurons)

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

    spikes = np.array(spikes)
    spikes = np.squeeze(spikes)

    return spikes, start_spikes, end_spikes


def run_raf_model(input_frequency, plot=False):
    num_samples = 10 if plot else 100

    input_frequencies = np.random.normal(input_frequency, 10, num_samples)
    input_frequencies.sort()

    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 15))
    spikes_at_frequency = []
    for f, ax in zip(input_frequencies, axes.flatten()):
        model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
        model.dT = dt
        timesteps = int(sim_time / model.dT)

        neuron_pop = model.add_neuron_population("Neuron", 1, rf_matlab_model, rf_params, rf_init)
        neuron_pop.spike_recording_enabled = True

        spike_times = generate_spike_times_frequency(0, 100, f)
        spike_times, start_spikes, end_spikes = set_input_frequency([0], [spike_times])

        input_pop = model.add_neuron_population("Input", 1, "SpikeSourceArray", {},
                                                {"startSpike": start_spikes, "endSpike": end_spikes})
        input_pop.set_extra_global_param("spikeTimes", spike_times)
        # input_pop.set_extra_global_param("spikeTimes", [0.0])
        input_pop.spike_recording_enabled = True

        model.add_synapse_population("InputNeuron", "SPARSE_GLOBALG", 0,
                                     input_pop, neuron_pop,
                                     "StaticPulse", {}, {"g": 50.0}, {}, {},
                                     "DeltaCurr", {}, {},
                                     init_connectivity("OneToOne", {}))

        model.build()
        model.load(num_recording_timesteps=timesteps)

        v_view = neuron_pop.vars["V"].view
        v = []
        while model.t < sim_time:
            model.step_time()
            neuron_pop.pull_var_from_device("V")
            v.append(np.copy(v_view[0]))

        model.pull_recording_buffers_from_device()

        input_spike_times, _ = input_pop.spike_recording_data
        neuron_spike_times, _ = neuron_pop.spike_recording_data
        spikes_at_frequency.append(len(neuron_spike_times))

        t = np.arange(0.0, 200.0, model.dT)
        ax.plot(t, v)
        ax.vlines(input_spike_times, ymin=-1, ymax=0, color="red", linestyle="--", label="input")
        ax.vlines(neuron_spike_times, ymin=0, ymax=1, color="blue", label="neuron")
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("V")
        ax.set_ylim([-1.1, 1.1])
        ax.set_title(f"Input frequency: {f:.2f} Hz, {len(neuron_spike_times)} spikes")
    ax.legend()
    fig.suptitle(f"RAF neuron centred around {input_frequency} Hz", fontsize=16)
    fig.tight_layout()
    if plot:
        fig.show()

    print('Number of spikes at each frequency:')
    print({f: s for f, s in zip(input_frequencies, spikes_at_frequency)})
    plt.plot(input_frequencies, spikes_at_frequency, "o")
    plt.xlim([np.min(input_frequencies), np.max(input_frequencies)])
    plt.title(f"Number of spikes of RAF neuron centred around {input_frequency} Hz")
    # plt.xticks(input_frequencies)
    plt.show()


sim_time = 200
dt = 0.01
omega = 300
input_frequency = 300

# RAF
rf_params = {"Damp": 0.1, "Omega": omega / 1000 * np.pi * 2}
rf_init = {"V": 0.0, "U": 0.0}

run_raf_model(input_frequency, plot=True)
run_raf_model(input_frequency, plot=False)
