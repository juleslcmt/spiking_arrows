import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/opt/spiking_arrows/')

import laser
from aestream import USBInput
import time
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Downscaled resolution
width = 640
height = 480

genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],
    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False)

# Resonate and fire neuron model
model = GeNNModel("float", "usb_genn")
#model.dT = 0.1

# Neuron parameters
filter_high_params = {"C": 1.0, "TauM": 0.1, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
filter_low_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 0.5, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -54.5,
                 'Ioffset': 0}
LIF_init = {'RefracTime': 0, 'V': -65}

# Network architecture
input_pop = model.add_neuron_population("input", width * height, genn_input_model, {}, {})
input_pop.set_extra_global_param("input", np.empty(9600, dtype=np.uint32))
input_pop.spike_recording_enabled = True

filter_high_pop = model.add_neuron_population("filter_high_pop", width * height, "LIF", filter_high_params, LIF_init)
filter_high_pop.spike_recording_enabled = True
filter_low_pop = model.add_neuron_population("filter_low_pop", width * height, "LIF", filter_low_params, LIF_init)
filter_low_pop.spike_recording_enabled = True

up_neuron = model.add_neuron_population("up_neuron", 1, "LIF", output_params, LIF_init)
down_neuron = model.add_neuron_population('down_neuron', 1, "LIF", output_params, LIF_init)
left_neuron = model.add_neuron_population('left_neuron', 1, "LIF", output_params, LIF_init)
right_neuron = model.add_neuron_population('right_neuron', 1, "LIF", output_params, LIF_init)
up_neuron.spike_recording_enabled = True
down_neuron.spike_recording_enabled = True
left_neuron.spike_recording_enabled = True
right_neuron.spike_recording_enabled = True

# Input to filter matrices
model.add_synapse_population("input_to_high_filter", "SPARSE_GLOBALG", 0,
                             input_pop, filter_high_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
model.add_synapse_population("input_to_low_filter", "SPARSE_GLOBALG", 0,
                             input_pop, filter_low_pop,
                             "StaticPulse", {}, {"g": 70.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))
                             
model.add_synapse_population("high_to_low", "SPARSE_GLOBALG", 0,
                             filter_high_pop, filter_low_pop,
                             "StaticPulse", {}, {"g": -1400.0}, {}, {},
                             "DeltaCurr", {}, {},
                             init_connectivity("OneToOne", {}))

# Weight matrices
max_weight_v = height*10
min_weight_v = height
max_weight_h = width*10
min_weight_h = width

height_up_weight_vector = np.linspace(max_weight_v, min_weight_v, height)
height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width,))

height_down_weight_vector = np.linspace(min_weight_v, max_weight_v, height)
height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width,))

width_left_weight_vector = np.linspace(max_weight_h, min_weight_h, width)
width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width,))

width_right_weight_vector = np.linspace(min_weight_h, max_weight_h, width)
width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width,))

# Filter high to directional neurons
model.add_synapse_population("excitatory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, up_neuron,
                             "StaticPulse", {}, {"g": height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, down_neuron,
                             "StaticPulse", {}, {"g": height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, left_neuron,
                             "StaticPulse", {}, {"g": width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("excitatory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_high_pop, right_neuron,
                             "StaticPulse", {}, {"g": width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
# Filter low to directional neurons
model.add_synapse_population("inhibitory_up_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, up_neuron,
                             "StaticPulse", {}, {"g": -1.25 * height_up_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_down_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, down_neuron,
                             "StaticPulse", {}, {"g": -1.25 * height_down_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_left_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, left_neuron,
                             "StaticPulse", {}, {"g": -1.25 * width_left_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})
                             
model.add_synapse_population("inhibitory_right_neuron", "DENSE_INDIVIDUALG", 0,
                             filter_low_pop, right_neuron,
                             "StaticPulse", {}, {"g": -1.25 * width_right_weight_matrix}, {}, {},
                             "DeltaCurr", {}, {})

# Build and simulate
model.build()
model.load(num_recording_timesteps=10)

with USBInput((height, width), device="genn") as stream:
    with laser.Laser() as l :
        l.on()
        state = (2000, 2000)

        while True:
            for i in range(10):
                stream.read_genn(input_pop.extra_global_params["input"].view)
                input_pop.push_extra_global_param_to_device("input")
                model.step_time()
            
            model.pull_recording_buffers_from_device()
            filter_high_spike_times, filter_high_spike_ids = filter_high_pop.spike_recording_data
            filter_low_spike_times, filter_low_spike_ids = filter_low_pop.spike_recording_data
            up_spike_times, up_spike_ids = up_neuron.spike_recording_data
            down_spike_times, down_spike_ids = down_neuron.spike_recording_data
            left_spike_times, left_spike_ids = left_neuron.spike_recording_data
            right_spike_times, right_spike_ids = right_neuron.spike_recording_data
            
            up_mov = len(up_spike_times)
            down_mov = len(down_spike_times)
            left_mov = len(left_spike_times)
            right_mov = len(right_spike_times)
            state = (2000-left_mov+right_mov, 2000+down_mov-up_mov)
            l.move(*state)



