import matplotlib.pyplot as plt
import numpy as np
import time
from pygenn.genn_model import GeNNModel, create_custom_neuron_class, init_connectivity

# Resolution
width = 64
height = 48

# Resonate and fire neuron model
model = GeNNModel("float", "rf", backend="SingleThreadedCPU")
model.dT = 0.1

# Neuron parameters
filter_high_params = {"C": 1.0, "TauM": 0.1, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
filter_low_params = {"C": 1.0, "TauM": 10.0, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -59.5, 'Ioffset': 0}
output_params = {"C": 1.0, "TauM": 0.5, "TauRefrac": 0.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -54.5,
                 'Ioffset': 0}
LIF_init = {'RefracTime': 0, 'V': -65}

# Setting frequencies and target position
simtime = 1000
target_frequency = 1000/10000
source_frequency = 1000/5000

start_spikes = [0 for i in range(height*width)]
end_spikes = [0 for i in range(height*width)]

target_cord_x = np.random.randint(width - 1)
target_cord_y = np.random.randint(height - 1)
spike_times = [np.arange(0, simtime, target_frequency)]
end_index = len(spike_times[0])
end_spikes[target_cord_x + target_cord_y * width] = end_index

# Source start position
source_cord_x = np.random.randint(width - 1)
source_cord_y = np.random.randint(height - 1)

start_time = 0
end_time = 10
new_spikes = np.arange(start_time, end_time, source_frequency)
spike_times.append(new_spikes)

start_index = end_index
end_index += len(new_spikes)
start_spikes[source_cord_x + source_cord_y * width] = start_index
end_spikes[source_cord_x + source_cord_y * width] = end_index

points = [[source_cord_x, source_cord_y, start_time, end_time]]

# Random walk
for i in range(99):
	decision = np.random.randint(3) # 0 -> Up, 1 -> Down, 2 -> Left, 3 -> Right
	
	if decision == 0 and source_cord_y != 0:
		source_cord_y += -1
	elif decision == 1 and source_cord_y != (height-1):
		source_cord_y += 1
	elif decision == 2 and source_cord_x != 0:
		source_cord_x += -1
	elif decision == 3 and source_cord_x != (width-1):
		source_cord_x += 1
		
	start_time = end_time
	end_time += 10
	new_spikes = np.arange(start_time, end_time, source_frequency)
	spike_times.append(new_spikes)

	start_index = end_index
	end_index += len(new_spikes)
	start_spikes[source_cord_x + source_cord_y * width] = start_index
	end_spikes[source_cord_x + source_cord_y * width] = end_index
	
	points.append([source_cord_x, source_cord_y, start_time, end_time])

# Build the network
input_pop = model.add_neuron_population("input_pop", height * width, "SpikeSourceArray", {},
                                        {"startSpike": start_spikes, "endSpike": end_spikes})
input_pop.set_extra_global_param("spikeTimes", np.concatenate(spike_times, axis=None))
input_pop.spike_recording_enabled = True

# Network architecture
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
model.load(num_recording_timesteps=100)

all_spikes = []
while model.t < simtime:
	model.step_time()
    
	if model.t % 10 == 0:
		model.pull_recording_buffers_from_device()

		filter_high_spike_times, filter_high_ids = filter_high_pop.spike_recording_data
		filter_low_spike_times, filter_low_ids = filter_low_pop.spike_recording_data
		up_spike_times, _ = up_neuron.spike_recording_data
		down_spike_times, _ = down_neuron.spike_recording_data
		left_spike_times, _ = left_neuron.spike_recording_data
		right_spike_times, _ = right_neuron.spike_recording_data

		all_spikes.append([filter_high_spike_times, filter_high_ids, filter_low_spike_times, filter_low_ids, up_spike_times, down_spike_times, left_spike_times, right_spike_times])

# --- Plotting the map and the spikes ---
plt.ion()

# Map
empty_x = np.tile(np.arange(0, width), height)
empty_y = np.tile([[j] for j in range(height)], width).flatten()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(empty_x, empty_y, 'o', markersize=2)
ax.plot(target_cord_x, target_cord_y, 'm*', markersize=8)
source_plot, = ax.plot(points[0][0], points[0][1], 'ro', markersize=5)

plt.yticks(np.arange(0, height))
plt.ylim(height, -1)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Map visualization')

# Filters and directional neurons
fig2 = plt.figure()

ax0 = fig2.add_subplot(611)
filter_high_plot, = ax0.plot(all_spikes[0][0], all_spikes[0][1], markersize=2)
ax0.set_xlim(0, simtime)
ax0.set_xlabel("time [ms]")
ax0.set_ylim((0, width*height))
ax0.set_title("Filter High")

ax1 = fig2.add_subplot(612)
filter_low_plot, = ax1.plot(all_spikes[0][2], all_spikes[0][3], markersize=4)
ax1.set_xlim(0, simtime)
ax1.set_xlabel("time [ms]")
ax1.set_ylim((0, width*height))
ax1.set_title("Filter Low")

ax2 = fig2.add_subplot(613)
up_plot = ax2.vlines(all_spikes[0][4], ymin=0, ymax=1, color="red", linestyle="--")
ax2.set_xlabel("time [ms]")
ax2.set_title("Up neuron")

ax3 = fig2.add_subplot(614)
down_plot = ax3.vlines(all_spikes[0][5], ymin=0, ymax=1, color="red", linestyle="--")
ax3.set_xlabel("time [ms]")
ax3.set_title("Down neuron")

ax4 = fig2.add_subplot(615)
left_plot = ax4.vlines(all_spikes[0][6], ymin=0, ymax=1, color="red", linestyle="--")
ax4.set_xlabel("time [ms]")
ax4.set_title("Left neuron")

ax5 = fig2.add_subplot(616)
right_plot = ax5.vlines(all_spikes[0][7], ymin=0, ymax=1, color="red", linestyle="--")
ax5.set_xlabel("time [ms]")
ax5.set_title("Right neuron")

for i in range(99):
	source_plot.set_xdata(points[i][0])
	source_plot.set_ydata(points[i][1])
	
	filter_high_plot.set_xdata(all_spikes[i][0])
	filter_high_plot.set_ydata(all_spikes[i][1])
	filter_low_plot.set_xdata(all_spikes[i][2])
	filter_low_plot.set_ydata(all_spikes[i][3])
	'''up_plot.set_xdata(all_spikes[i][4])
	down_plot.set_xdata(all_spikes[i][5])
	left_plot.set_xdata(all_spikes[i][6])
	right_plot.set_xdata(all_spikes[i][7])'''
	
	fig.canvas.draw()
	fig.canvas.flush_events()

	time.sleep(0.1)