import numpy as np


def generate_spike_times_frequency(start_time, end_time, frequency):
    end_time /= 1000

    sample_rate = 100000
    amplitude = 1
    theta = 0

    time = np.arange(start_time, end_time, 1 / sample_rate)
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    # local min
    maxima = (np.diff(np.sign(np.diff(sinewave))) < 0).nonzero()[0] + 1  # local max

    return list(time[maxima] * 1000)


def set_input_frequency(spiking_neurons, spike_times, num_neurons):
    num_neurons = num_neurons

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

    # spikes = np.array(spikes)
    # spikes = np.squeeze(spikes)

    return spikes, start_spikes, end_spikes


def raf_model():
    from pygenn.genn_model import create_custom_neuron_class

    return create_custom_neuron_class(
        "RF",
        param_names=["Damp", "Omega", 'Vspike'],
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
    
        if ( $(V) > $(Vspike) ) {
        $(V) = $(Vspike);
        }
        """,
        threshold_condition_code=
        """
        $(V) >= $(Vspike) - 0.01
        """,
        reset_code=
        """
        """)
