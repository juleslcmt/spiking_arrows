import matplotlib.pyplot as plt
from pygenn.genn_model import GeNNModel, init_connectivity

from helper_functions import *


def run_raf_model(input_frequency, plot=False):
    num_samples = 10 if plot else 100

    input_frequencies = np.random.normal(input_frequency, 10, num_samples)
    input_frequencies.sort()

    fig, axes = plt.subplots(num_samples // 2, 2, figsize=(15, 15))
    spikes_at_frequency = []
    for f, ax in zip(input_frequencies, axes.flatten()):
        model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
        model.dT = dt
        timesteps = int(sim_time / model.dT)

        neuron_pop = model.add_neuron_population("Neuron", 1, raf_model(), rf_params, rf_init)
        neuron_pop.spike_recording_enabled = True

        spike_times = generate_spike_times_frequency(0, 100, f)
        spike_times, start_spikes, end_spikes = set_input_frequency([0], [spike_times], 1)

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
    fig, ax = plt.subplots()
    ax.plot(input_frequencies, spikes_at_frequency, "o")
    ax.set_xlim([np.min(input_frequencies), np.max(input_frequencies)])
    ax.set_title(f"Number of spikes of RAF neuron centred around {input_frequency} Hz")
    # plt.xticks(input_frequencies)
    fig.show()


sim_time = 200
dt = 0.01
omega = 300
input_frequency = 300

# RAF
rf_params = {"Damp": 0.1, "Omega": omega / 1000 * np.pi * 2}
rf_init = {"V": 0.0, "U": 0.0}

run_raf_model(input_frequency, plot=True)
run_raf_model(input_frequency, plot=False)
