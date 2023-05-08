import math
from threading import Thread

import numpy as np
import pyNN.brian2 as sim
#import spynnaker8 as sim
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0}
excitatory_params = {"cm": 1.0, "tau_m": 100.0, "tau_refrac": 0.0, "v_rest": -65.0, "v_thresh": -59.5}
inhibitory_params = {"cm": 1.0, "tau_m": 100.0, "tau_refrac": 0.0, "v_rest": -65.0, "v_thresh": -59.5}
output_params = {"cm": 1.0, "tau_m": 10.0, "tau_refrac": 0.0, "tau_syn_E": 1.0, "tau_syn_I": 1.0, "v_rest": -65.0,
                 "v_reset": -65.0, "v_thresh": -64.75}

# Camera resolution
width = 8
height = 6

# Variables for the simulation
simtime = 160.0
extension = simtime * 0.25
pos_change = 40.0
directional_spikes = []
all_points = []


def build_network():
    global directional_spikes

    # --- Simulation ---
    sim.setup(global_params["min_delay"])

    # -- Network architecture --
    # - Populations -
    spike_sources = []
    start_time = 0
    end_time = pos_change
    for i in range(0, 1 + int(simtime / pos_change)):
        spike_sources.append(sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(start_time, end_time, 1.0))))
        start_time = end_time
        end_time += pos_change

    excitatory_matrix = sim.Population(width * height, sim.IF_curr_exp(**excitatory_params),
                                       initial_values={'v': excitatory_params["v_rest"]},
                                       label="excitatory_matrix")
    inhibitory_matrix = sim.Population(width * height, sim.IF_curr_exp(**inhibitory_params),
                                       initial_values={'v': inhibitory_params["v_rest"]},
                                       label="inhibitory_matrix")

    # Up (0), Down (1), Right (2), Left (3)
    directional_layer = sim.Population(4, sim.IF_curr_exp(**output_params),
                                       initial_values={'v': output_params["v_rest"]},
                                       label="output_neuron")

    # - Weight matrices -
    height_up_weight_vector = np.linspace(1, 0, height)
    height_up_weight_matrix = np.tile(height_up_weight_vector, (width, 1)).T.reshape((height * width, 1))

    height_down_weight_vector = np.linspace(0, 1, height)
    height_down_weight_matrix = np.tile(height_down_weight_vector, (width, 1)).T.reshape((height * width, 1))

    width_right_weight_vector = np.linspace(0, 1, width)
    width_right_weight_matrix = np.tile(width_right_weight_vector, (1, height)).T.reshape((height * width, 1))

    width_left_weight_vector = np.linspace(1, 0, width)
    width_left_weight_matrix = np.tile(width_left_weight_vector, (1, height)).T.reshape((height * width, 1))

    # --- Test points ---
    target_cord_x = np.random.randint(width - 1)
    target_cord_y = np.random.randint(height - 1)
    for i in range(0, int(simtime / pos_change)):
        source_cord_x = np.random.randint(width - 1)
        source_cord_y = np.random.randint(height - 1)

        pair_points = [target_cord_x, target_cord_y, source_cord_x, source_cord_y]
        all_points.append(pair_points)

        # - Connections -
        sim.Projection(spike_sources[i], excitatory_matrix.__getitem__([target_cord_y * width + target_cord_x]),
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"]))
        sim.Projection(spike_sources[i], inhibitory_matrix.__getitem__([source_cord_y * width + source_cord_x]),
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"]))

    # EXCITATORY
    # Up neuron
    sim.Projection(excitatory_matrix, directional_layer.__getitem__([0]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=height_up_weight_matrix, delay=global_params["min_delay"]),
                   receptor_type="excitatory")

    # Down neuron
    sim.Projection(excitatory_matrix, directional_layer.__getitem__([1]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=height_down_weight_matrix, delay=global_params["min_delay"]),
                   receptor_type="excitatory")

    # Right
    sim.Projection(excitatory_matrix, directional_layer.__getitem__([2]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=width_right_weight_matrix, delay=global_params["min_delay"]),
                   receptor_type="excitatory")

    # Left
    sim.Projection(excitatory_matrix, directional_layer.__getitem__([3]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=width_left_weight_matrix, delay=global_params["min_delay"]),
                   receptor_type="excitatory")

    # INHIBITORY
    # Up neuron
    sim.Projection(inhibitory_matrix, directional_layer.__getitem__([0]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=height_up_weight_matrix * -1, delay=global_params["min_delay"]),
                   receptor_type="inhibitory")

    # Down neuron
    sim.Projection(inhibitory_matrix, directional_layer.__getitem__([1]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=height_down_weight_matrix * -1, delay=global_params["min_delay"]),
                   receptor_type="inhibitory")

    # Right
    sim.Projection(inhibitory_matrix, directional_layer.__getitem__([2]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=width_right_weight_matrix * -1, delay=global_params["min_delay"]),
                   receptor_type="inhibitory")

    # Left
    sim.Projection(inhibitory_matrix, directional_layer.__getitem__([3]), sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=width_left_weight_matrix * -1, delay=global_params["min_delay"]),
                   receptor_type="inhibitory")

    # -- Recording --
    directional_layer.record(["spikes"])

    # -- Run simulation --
    sim.run(simtime)
    sim.run(extension)

    # -- Get data from the simulation --
    directional_spikes = directional_layer.get_data(variables=["spikes"]).segments[0].spiketrains

    # - End simulation -
    sim.end()


def plot_results():
    n_points = int(simtime / pos_change)
    n_points_sqrt = math.ceil(math.sqrt(n_points))

    plt.figure()
    for i in range(0, n_points):
        plt.subplot(n_points_sqrt, n_points_sqrt, i + 1)

        plt.plot(np.tile(np.arange(0, width), height),
                 np.tile([[j] for j in range(height)], width).flatten(),
                 'o', markersize=3)
        plt.plot(all_points[i][0], all_points[i][1], 'm*', markersize=8)
        plt.plot(all_points[i][2], all_points[i][3], 'ro', markersize=5)
        plt.yticks(np.arange(0, height))
        plt.ylim(height, -1)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Matrix visualization - Pos. ' + str(i))
    plt.tight_layout()

    plt.figure()
    plt.plot(directional_spikes[0], [0] * len(directional_spikes[0]), 'ro', markersize=5)
    plt.plot(directional_spikes[1], [1] * len(directional_spikes[1]), 'ro', markersize=5)
    plt.plot(directional_spikes[2], [2] * len(directional_spikes[2]), 'ro', markersize=5)
    plt.plot(directional_spikes[3], [3] * len(directional_spikes[3]), 'ro', markersize=5)
    plt.title('Directional neuron responses')
    plt.xlabel('Time (ms)')
    plt.xlim([0, int(simtime + extension)])
    plt.ylabel('Direction')
    plt.gca().set_yticks([0, 1, 2, 3])
    plt.gca().set_yticklabels(['Up', 'Down', 'Right', 'Left'])
    label = mpatches.Patch(color='red', label='Spike')
    plt.legend(handles=[label])
    plt.show()


if __name__ == '__main__':
    # Building the network
    build_network()

    # Plotting maps
    plot_results()
