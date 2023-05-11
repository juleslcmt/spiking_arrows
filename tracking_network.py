import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/opt/spiking_arrows/')

import laser
from aestream import USBInput
import time
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

check_time = 0.001 
dt = 0.01

# Sensor resolution
width = 640
height = 480

# DVS Sensor
genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],
    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False)

# Resonate and fire neuron model
rf_model = create_custom_neuron_class(
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

model = GeNNModel("float", "tracking_network")
model.dT = dt
sim_time = 200
timesteps = int(sim_time / model.dT)

omega = 300
rf_params = {"Damp": 0.1, "Omega": omega / 1000 * np.pi * 2}
rf_init = {"V": 0.0, "U": 0.0}

# Neuron parameters
filter_high_params = {"C": 1.0, "TauM": 1.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -57.0, 'Ioffset': 0}
filter_low_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.95,
                 'Ioffset': 0}

# Network architecture

input_pop = model.add_neuron_population("input", width * height, genn_input_model, {}, {})
input_pop.set_extra_global_param("input", np.empty(9600, dtype=np.uint32))
input_pop.spike_recording_enabled = True

# WE NEED PARAMETER TUNING FOR THE TWO RF POPULATIONS

src_neuron_pop = model.add_neuron_population("RF_src_neurons", width*height, rf_model, rf_params, rf_init)
src_neuron_pop.spike_recording_enabled = True

tgt_neuron_pop = model.add_neuron_population("RF_tgt_neurons", width*height, rf_model, rf_params, rf_init)
tgt_neuron_pop.spike_recording_enabled = True

up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, LIF_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, LIF_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, LIF_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, LIF_init)
up_neuron.spike_recording_enabled = True
down_neuron.spike_recording_enabled = True
left_neuron.spike_recording_enabled = True
right_neuron.spike_recording_enabled = True

model.add_synapse_population("InputNeuron", "SPARSE_GLOBALG", 0,
                                     input_pop, src_neuron_pop,
                                     "StaticPulse", {}, {"g": 50.0}, {}, {},
                                     "DeltaCurr", {}, {},
                                     init_connectivity("OneToOne", {}))

model.add_synapse_population("InputNeuron", "SPARSE_GLOBALG", 0,
                                     input_pop, tgt_neuron_pop,
                                     "StaticPulse", {}, {"g": 50.0}, {}, {},
                                     "DeltaCurr", {}, {},
                                     init_connectivity("OneToOne", {}))

model.add_synapse_population("high_to_low", "SPARSE_INDIVIDUALG", 0,
                             src_neuron_pop, tgt_neuron_pop,
                             "StaticPulse", {}, {"g": -1400.0}, {}, {},
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
                             src_neuron_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, down_neuron,
                             "StaticPulse", {}, {"g": -1 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, left_neuron,
                             "StaticPulse", {}, {"g": -1 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, right_neuron,
                             "StaticPulse", {}, {"g": -1 * width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=10)

v_view_up = up_neuron.vars["V"].view
v_view_down = down_neuron.vars["V"].view
v_view_left = left_neuron.vars["V"].view
v_view_right = right_neuron.vars["V"].view

#spikes_history = input_pop.vars["current_spikes"].view

v_all = [v_view_up, v_view_down, v_view_left, v_view_right]

step_size = 100 
arrow_neurons = [up_neuron,down_neuron,left_neuron,right_neuron]
moves = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]

with USBInput((height, width), device="genn") as stream:
    with laser.Laser() as l :
        l.on()
        state = (2000, 2000)

        time_start = time.time()
        # Loop forever
        while True:
            #print(stream)
            for i in range(10):
                stream.read_genn(input_pop.extra_global_params["input"].view)
                #print("1ms time window with ", np.count_nonzero(input_pop.extra_global_params["input"].view), " events")

                #print(input_pop.extra_global_params["input"].view)
                #input_pop.push_extra_global_param_to_device("input")
                input_pop.push_extra_global_param_to_device("input")
                model.step_time()
                time.sleep(check_time)
                moving = False
                #print(model.pull_prev_spikes_from_device("input"))
                #model.pull_recording_buffers_from_device()
                #print(model.pull_current_spikes_from_device("input"))

                #print(model.pull_current_spike_events_from_device("input"))
                #print(model.pull_spike_events_from_device("input"))
                #print(model.pull_spikes_times_from_device("input"))
                """ for j,neuron in enumerate(arrow_neurons) :
                if model.pull_spikes_from_device(neuron) :
                    print("spiking " + neuron)
                    state = (max(0,min(4095,state[0]+moves[j][0])), max(0,min(4095,state[1]+moves[j][1])))
                    moving = True
            
            if moving :
                print(state)
                l.move(*state) """
            
            model.pull_recording_buffers_from_device()
            """ 
            spike_times, spike_ids = input_pop.spike_recording_data
            spike_x = spike_ids % 640
            spike_y = spike_ids // 640
            print(spike_x)
            print(spike_y)
            print(spike_times) """
            spike_times, spike_ids = src_neuron_pop.spike_recording_data
            spike_x = spike_ids % 640
            spike_y = spike_ids // 640
            """ print(spike_x)
            print(spike_y)
            print(spike_times) """
            """ spike_times, spike_ids = filter_low_pop.spike_recording_data
            spike_x = spike_ids % 640
            spike_y = spike_ids // 640
            print(spike_x)
            print(spike_y)
            print(spike_times)
            spike_times, spike_ids = up_neuron.spike_recording_data

            spike_x = spike_ids % 640
            spike_y = spike_ids // 640
            print(spike_x)
            print(spike_y)
            print(spike_times) """
            up_times, up_ids = up_neuron.spike_recording_data
            for j,neuron in enumerate(arrow_neurons) :
                s_times, s_ids = neuron.spike_recording_data
                if len(s_ids) :
                    print("spiking ")
                    state = (max(0,min(4095,state[0]+moves[j][0])), max(0,min(4095,state[1]+moves[j][1])))
                    moving = True
            if moving :
                print(state)
                l.move(*state)
