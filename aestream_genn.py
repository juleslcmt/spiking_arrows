import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/opt/spiking_arrows/')

import laser
from aestream import USBInput
import time
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Simulation variables
check_time = 0.001  # Check output spikes each 1ms to sample correctly 100Hz(source) and 300Hz (target) signals

# Downscaled resolution
width = 480
height = 640

genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],
    
    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False)


model = GeNNModel("float", "usb_genn")
DVS_pop = model.add_neuron_population("input", width*height, genn_input_model, {}, {})
DVS_pop.set_extra_global_param("input", np.empty(9600, dtype=np.uint32))
DVS_pop.spike_recording_enabled = True

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

model.dT = check_time
rf_excitatory_params = {"Damp": 2.5, "Omega": 2.0 * np.pi * 2.0}
rf_inhibitory_params = {"Damp": 2.5, "Omega": 3.0 * np.pi * 2.0}
rf_init = {"V": 0.0, "U": 0.0}
output_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -64.75,
                 'Ioffset': 0}
output_init = {'RefracTime': 0, 'V': -65}

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
                             DVS_pop, excitatory_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
model.add_synapse_population("input_inhibitory", "SPARSE_GLOBALG", 0,
                             DVS_pop, inhibitory_pop,
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

model.build()
timesteps = 10
model.load(num_recording_timesteps=10)

v_view_up = up_neuron.vars["V"].view
v_view_down = down_neuron.vars["V"].view
v_view_left = left_neuron.vars["V"].view
v_view_right = right_neuron.vars["V"].view
v_all = [v_view_up, v_view_down, v_view_left, v_view_right]

excitatory_view = excitatory_pop.vars["V"].view
inhibitory_view = inhibitory_pop.vars["V"].view

step_size = 100 
arrow_neurons = ["up_neuron","down_neuron","left_neuron","right_neuron"]
moves = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]

# Connect to a USB camera, receiving tensors of shape (640, 480)
# By default, we send the tensors to the CPU
#   - if you have a GPU, try changing this to "cuda"
with USBInput((height, width), device="genn") as stream:
    with laser.Laser() as l :
        l.on()
        state = (2000, 2000)

        time_start = time.time()
        # Loop forever
        while True:
            #print(stream)
            for i in range(1000000):
                stream.read_genn(DVS_pop.extra_global_params["input"].view)
                #print("1ms time window with ", np.count_nonzero(DVS_pop.extra_global_params["input"].view), " events")
                DVS_pop.push_extra_global_param_to_device("input")
            
                model.step_time()
                up_neuron.pull_var_from_device("V")
                down_neuron.pull_var_from_device("V")
                left_neuron.pull_var_from_device("V")
                right_neuron.pull_var_from_device("V")

                excitatory_pop.pull_var_from_device("V")
                inhibitory_pop.pull_var_from_device("V")
                time.sleep(check_time)

                moving = False
                print("step")
                for j,neuron in enumerate(arrow_neurons) :
                    print(neuron)
                    if model.pull_spikes_from_device(neuron) :
                        print("spiking " + neuron)
                        state = (max(0,min(4095,state[0]+moves[j][0])), max(0,min(4095,state[1]+moves[j][1])))
                        moving = True
                
                if moving :
                    print(state)
                    l.move(*state)
            #excitatory_spike_times, excitatory_ids = excitatory_pop.spike_recording_data
            #inhibitory_spike_times, inhibitory_ids = inhibitory_pop.spike_recording_data
            #up_spike_times, _ = up_neuron.spike_recording_data
            spike_times, spike_ids = DVS_pop.spike_recording_data

            # Plotting code from Jamie (GeNN)
            #fig, axis = plt.subplots()
            spike_x = spike_ids % height
            spike_y = spike_ids // height
            
            #axis.scatter(spike_x, spike_y, s=1)
            #plt.show()

    
