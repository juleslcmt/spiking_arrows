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
"""
def rk4_step( x, t, f, dt, args ):
    const scalar k1 = f( t, x, *args )
    const scalar k2 = f( t + dt / 2, x + dt * k1 / 2, *args )
    const scalar k3 = f( t + dt / 2, x + dt * k2 / 2, *args )
    const scalar k4 = f( t + dt, x + dt * k3, *args )
    
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    """
izi_rz = create_custom_neuron_class(
    'IzhikevichBio',
    param_names=['a', 'b', 'c', 'd', 'k', 'Cm', 'Vr', 'Vth', 'Vspike'],
    var_name_types=[('V', 'scalar'), ('U', 'scalar')],
    support_code=
    """
        SUPPORT_CODE_FUNC scalar dvdt( scalar V, scalar k, scalar Vr, scalar Vth, scalar Isyn, scalar U){
        return k * ( V - Vr ) * ( V - Vth ) - U + Isyn;
    }
    """,
    sim_code=
    """
    const scalar k1 = dvdt( $(V), $(k), $(Vr), $(Vth), $(Isyn), $(U) );
    const scalar k2 = dvdt( $(V) + DT * k1 / 2, $(k), $(Vr), $(Vth), $(Isyn), $(U) );
    const scalar k3 = dvdt( $(V) + DT * k2 / 2, $(k), $(Vr), $(Vth), $(Isyn), $(U) );
    const scalar k4 = dvdt( $(V) + DT * k3, $(k), $(Vr), $(Vth), $(Isyn), $(U) );
    $(V) += DT * ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6;
    
    $(U) += $(a) * ( $(b) * ( $(V) - $(Vr) ) - $(U) ) * DT;
    
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
    $(V) = $(c);
    $(U) += $(d);
    """)

model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
sim_time = 200
model.dT = 0.01  # 0.1 ms timestep
timesteps = int(sim_time / model.dT)

# RAF
omega = 100
rf_params = {"Damp": 0.1, "Omega": omega / 1000 * np.pi * 2}
rf_init = {"V": 0.0, "U": 0.0}
neuron_pop = model.add_neuron_population("Neuron", 1, rf_matlab_model, rf_params, rf_init)

# IZHIKEVICH RZ
izk_param = {
    "a": 1.04,
    "b": 0.996,
    'Cm': 0.05,
    'k': 0.11,
    'Vr': -60,
    "c": -55,
    "d": 10,
    "Vspike": 50
}
izk_vars = {
    "V": -65.0,
    "U": -20.0
}
izk_param['Vth'] = izk_param['Vr'] - izk_param['Cm'] / izk_param['k']
# neuron_pop = model.add_neuron_population("Neuron", 1, izi_rz, izk_param, izk_vars)
neuron_pop.spike_recording_enabled = True


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


spike_times = generate_spike_times_frequency(0, 100, 75)
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

timesteps = np.arange(0.0, 200.0, model.dT)

# Create figure with 4 axes
fig, axis = plt.subplots(figsize=(12, 8))
axis.plot(timesteps, v)
axis.vlines(input_spike_times, ymin=-1, ymax=0, color="red", linestyle="--", label="input")
axis.vlines(neuron_spike_times, ymin=0, ymax=1, color="blue", label="neuron")
axis.set_xlabel("time [ms]")
axis.set_ylabel("V")
axis.set_ylim([-1.1, 1.1])
axis.legend()
fig.show()
