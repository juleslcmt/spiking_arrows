import matplotlib
import laser
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from aestream import USBInput
from pygenn.genn_model import GeNNModel
from pygenn.genn_model import create_custom_neuron_class

genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],

    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False)


model = GeNNModel("float", "usb_genn", backend="SingleThreadedCPU")
pop = model.add_neuron_population("input", 640 * 480, genn_input_model, {}, {})
pop.set_extra_global_param("input", np.empty(9600, dtype=np.uint32))
pop.spike_recording_enabled = True

model.build()
model.load(num_recording_timesteps=100)

moves = {"w": (-100, 0), "s": (100, 0), "a": (0, -100), "d": (0, 100)}

# Connect to a USB camera, receiving tensors of shape (640, 480)
# By default, we send the tensors to the CPU
#   - if you have a GPU, try changing this to "cuda"
with USBInput((640, 480), device="genn") as stream:
    # Loop forever
    with laser.Laser() as l :
        l.on()
        state = (2000, 2000)
        while True:
            """ for i in range(100):
                stream.read_genn(pop.extra_global_params["input"].view)
                pop.push_extra_global_param_to_device("input")

                model.step_time()
            
            #extracting spike arrows
            model.pull_recording_buffers_from_device()
            spike_times, spike_ids = pop.spike_recording_data """
            #4 spikes ids and a given number of spike times

            l.blink(200) #the time to turn off is approximately 50ms
            if state == None :
                state = (4095, 4095)
                state = (state[0] + move[spike_ids])
            """ fig, axis = plt.subplots()
            spike_x = spike_ids % 640
            spike_y = spike_ids // 640
            axis.scatter(spike_x, spike_y, s=1)
            plt.show() """

            #moving the laser
            state = (min(4095, max(0, state[0])), min(4095, max(0, state[1])))
            l.move(*state)