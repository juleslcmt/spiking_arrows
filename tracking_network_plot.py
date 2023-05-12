import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/opt/spiking_arrows/')

import laser
from aestream import USBInput
from helper_functions import raf_model
import time
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

check_time = 0.001 
dt = 0.001

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



model = GeNNModel("float", "tracking_network")
model.dT = dt
#sim_time = 200
#timesteps = int(sim_time / model.dT)

omega_tgt = 300
omega_src = 20
rf_params_tgt = {"Damp": 0.1, "Omega": omega_tgt / 1000 * np.pi * 2, "Vspike" : 0.9}
rf_params_src = {"Damp": 0.1, "Omega": omega_src / 1000 * np.pi * 2, "Vspike" : 0.9}
rf_init = {"V": 0.0, "U": 0.0}
LIF_init = {'RefracTime': 0, 'V': -65}

# Neuron parameters
filter_high_params = {"C": 1.0, "TauM": 1.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.95, 'Ioffset': 0}
filter_low_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -20, 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.95,
                 'Ioffset': 0}

# Network architecture

input_pop = model.add_neuron_population("input", width * height, genn_input_model, {}, {})
input_pop.set_extra_global_param("input", np.empty(614400, dtype=np.uint32))
input_pop.spike_recording_enabled = True

# WE NEED PARAMETER TUNING FOR THE TWO RF POPULATIONS

src_neuron_pop = model.add_neuron_population("RF_src_neurons", width*height, raf_model(), rf_params_src, rf_init)
src_neuron_pop.spike_recording_enabled = True

tgt_neuron_pop = model.add_neuron_population("RF_tgt_neurons", width*height, raf_model(), rf_params_tgt, rf_init)
tgt_neuron_pop.spike_recording_enabled = True

up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, LIF_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, LIF_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, LIF_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, LIF_init)
up_neuron.spike_recording_enabled = True
down_neuron.spike_recording_enabled = True
left_neuron.spike_recording_enabled = True
right_neuron.spike_recording_enabled = True

model.add_synapse_population("DVSsrc", "SPARSE_GLOBALG", 0,
                                     input_pop, src_neuron_pop,
                                     "StaticPulse", {}, {"g": 50.0}, {}, {},
                                     "DeltaCurr", {}, {},
                                     init_connectivity("OneToOne", {}))

model.add_synapse_population("DVStgt", "SPARSE_GLOBALG", 0,
                                     input_pop, tgt_neuron_pop,
                                     "StaticPulse", {}, {"g": 50.0}, {}, {},
                                     "DeltaCurr", {}, {},
                                     init_connectivity("OneToOne", {}))

model.add_synapse_population("high_to_low", "SPARSE_GLOBALG", 0,
                                    tgt_neuron_pop, src_neuron_pop,
                                    "StaticPulse", {}, {"g": -1400.0}, {}, {},
                                    "DeltaCurr", {}, {},
                                    init_connectivity("OneToOne", {}))
                        
# Weight matrices
height_up_weight_vector = np.linspace(0, 1, height) #np.linspace(1, 0, height) #the vertical axis is inverted
height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width,))
height_down_weight_vector = np.linspace(1, 0, height) #np.linspace(0, 1, height) #the vertical axis is inverted
height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width,))
width_right_weight_vector = np.linspace(0, 1, width)
width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width,))
width_left_weight_vector = np.linspace(1, 0, width)
width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width,))

# Excitatory and inhibitory matrices to directional neurons
model.add_synapse_population("excitatory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, up_neuron,
                             "StaticPulse", {}, {"g": -0.2 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, down_neuron,
                             "StaticPulse", {}, {"g": -0.2 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, left_neuron,
                             "StaticPulse", {}, {"g": -0.2 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             tgt_neuron_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             src_neuron_pop, right_neuron,
                             "StaticPulse", {}, {"g": -0.2 * width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=10)

v_view_up = up_neuron.vars["V"].view
v_view_down = down_neuron.vars["V"].view
v_view_left = left_neuron.vars["V"].view
v_view_right = right_neuron.vars["V"].view

v_all = [v_view_up, v_view_down, v_view_left, v_view_right]

step_size = 100
arrow_neurons = [up_neuron,down_neuron,left_neuron,right_neuron]
moves = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]
total_high_ids = []
total_low_ids = []
plotting = True

with USBInput((width, height), device="genn") as stream:
    with laser.Laser() as l :
        l.on()
        state = (2000, 2000)
        l.blink(50)
        l.move(*state)
        time_start = time.time()
        # Loop forever
        while True:
            for i in range(10):
                stream.read_genn(input_pop.extra_global_params["input"].view)
                #print(input_pop.extra_global_params["input"].view[:20])
                input_pop.push_extra_global_param_to_device("input")
                model.step_time()
                #print("sim_time : ", model.t)
                #print("real time : ", time.time() - time_start)
                delay = model.t - (time.time() - time_start)
                if delay > 0 :
                    time.sleep(delay)
            moving = False
            
            model.pull_recording_buffers_from_device()
            #spike_times, spike_ids = src_neuron_pop.spike_recording_data
            #spike_x = spike_ids % 640
            #spike_y = spike_ids // 640

            #up_times, up_ids = up_neuron.spike_recording_data
            for j, neuron in enumerate(arrow_neurons) :
                s_times, s_ids = neuron.spike_recording_data
                if len(s_ids) :
                    print(moves[j])
                    state = (max(0,min(4095,state[0]+moves[j][0])), max(0,min(4095,state[1]+moves[j][1])))
                    #print(state)
                    moving = True
            if moving :
                l.move(*state)

            #if (time.time() - time_start) > 5 :
            if plotting :
                high_times, high_ids = tgt_neuron_pop.spike_recording_data
                low_times, low_ids = src_neuron_pop.spike_recording_data
                total_low_ids = np.concatenate((total_low_ids, low_ids))
                total_high_ids = np.concatenate((total_high_ids, high_ids))
                if time.time() - time_start > 5 :

                    print("Plotting")
                    fig, axis = plt.subplots(2,1)
                    spike_x = total_high_ids % width
                    spike_y = total_high_ids // width
                    low_x = total_low_ids % width
                    low_y = total_low_ids // width
                    axis[0].scatter(spike_x, spike_y, s=1, c='green')
                    axis[1].scatter(low_x, low_y, s=1, c='red')
                    plt.show()
                    exit()
