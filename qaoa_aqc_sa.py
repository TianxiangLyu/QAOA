import matplotlib.pyplot as plt

import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from qiskit_aer import AerSimulator
# Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_distribution

np.random.seed(2)
nqubits = 9  # Number of nodes in graph
#nqubits = 4
n_edge = 17
graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, nqubits, 1))
edge_list = []
ep = np.random.randint(low=0, high=nqubits, size=2*n_edge, dtype=int)
#for edge in range(n_edge):
    #edge_list.append(((ep[2*edge]), ep[2*edge+1], 1.0))
#for edge in range(n_edge):
#    edge_list.append(((0), ep[2*edge+1], 1.0))
for i in range(1, nqubits):
    edge_list.append(((0), i, 1.0))
#print(edge_list)
edge_list = list(set(edge_list))
edge_list = [edge for edge in edge_list if edge[0] != edge[1]]
#edge_list = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
#print(edge_list)
n_edge = len(edge_list)
graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)
plt.savefig('graph.png')
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
backend = AerSimulator()
backend.name
target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
qc_qaoa = QuantumCircuit(nqubits)
T = 16000
qc_0 = QuantumCircuit(nqubits)
for i in range(nqubits):
    qc_0.h(i)
qc_qaoa.append(qc_0, [i for i in range(nqubits)])
for i in range(0,T+1):
    beta = (1 - i/T)
    qc_mix = QuantumCircuit(nqubits)
    for k in range(nqubits):
        qc_mix.rx(2 * beta, k)
    gamma = (i/T)
    qc_p = QuantumCircuit(nqubits)
    for edge in range(len(edge_list)):  # pairs of nodes
        qc_p.rzz(2 * gamma, edge_list[edge][0], edge_list[edge][1])
        qc_p.barrier()
    qc_qaoa.append(qc_mix, [i for i in range(0, nqubits)])
    qc_qaoa.append(qc_p, [i for i in range(0, nqubits)])
qc_qaoa.measure_all()
qc_isa = pm.run(qc_qaoa)
#qc_isa.decompose().draw(output="mpl")
#plt.savefig('qaoa_aqc.png')
session = Session(backend=backend)
sampler = Sampler(mode=session)
result = sampler.run([qc_isa]).result()
samp_dist = result[0].data.meas.get_counts()
session.close()
plot_distribution(samp_dist)
plt.show()
counts_int = result[0].data.meas.get_int_counts()
counts_bin = result[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
print(final_distribution_int)
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
print("most_likely: ", most_likely)
most_likely_bitstring = to_bitstring(most_likely, len(graph))
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)
#plt.savefig('qaoa_aqc.png')
#plt.show()

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 10})
final_bits = final_distribution_bin
values = np.abs(list(final_bits.values()))
top_4_values = sorted(values, reverse=True)[:4]
positions = []
for value in top_4_values:
    positions.append(np.where(values == value)[0])
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1)
plt.xticks(rotation=45)
plt.title("Result Distribution")
plt.xlabel("Bitstrings (reversed)")
plt.ylabel("Probability")
ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
for p in positions:
    ax.get_children()[p[0]].set_color("tab:purple")
plt.show()
# auxiliary function to plot graphs
def plot_result(G, x):
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    pos, default_axes = rx.spring_layout(G), plt.axes(frameon=True)
    rx.visualization.mpl_draw(G, node_color=colors, node_size=100, alpha=0.8, pos=pos)

#plot_result(graph, most_likely_bitstring)

import random
import math
import networkx as nx
import numpy as np
# Function to compute the cut value (sum of weights of edges between two sets)
def cut_value(graph, partition):
    cut = 0
    for u, v, data in graph.edges(data=True):
        if partition[u] != partition[v]:
            cut += data.get("weight", 1)  # Default weight is 1 if not provided
    return cut

# Function to flip a random node's partition
def flip_partition(partition, node):
    new_partition = partition.copy()
    new_partition[node] = 1 - new_partition[node]
    return new_partition

# Simulated Annealing algorithm
def simulated_annealing_maxcut(graph, initial_temp, final_temp, alpha, max_steps, init_partition):
    # Initialize random partition
    partition = init_partition
    best_partition = partition
    current_cut_value = cut_value(graph, partition)
    best_cut_value = current_cut_value
    temp = initial_temp

    for step in range(max_steps):
        if temp <= final_temp:
            break
        # Pick a random node and flip its partition
        node = random.choice(list(graph.nodes()))
        new_partition = flip_partition(partition, node)
        new_cut_value = cut_value(graph, new_partition)
        # Calculate the difference in cut values
        delta = new_cut_value - current_cut_value
        # Accept new partition if it's better or with a probability (simulated annealing step)
        if delta > 0 or random.random() < math.exp(delta / temp):
            partition = new_partition
            current_cut_value = new_cut_value
            # Update the best partition found
            if current_cut_value > best_cut_value:
                best_partition = partition
                best_cut_value = current_cut_value
        # Cool down the temperature
        temp *= alpha
    return best_partition, best_cut_value
# Example usage
    # Create a sample graph
G = nx.Graph()
G.add_weighted_edges_from(edge_list)
# Parameters for simulated annealing
initial_temp = 100
final_temp = 1
alpha = 0.99999
max_steps = 100000
#init_partition = {node: random.choice([0, 1]) for node in G.nodes()}
init_partition = {}
for i in range(nqubits):
    init_partition[np.int64(i)] = most_likely_bitstring[i]
#print(init_partition)
# Solve Max-Cut using simulated annealing
init_cut_value = cut_value(G, init_partition)
print("QAOA cut value:", init_cut_value)
best_partition, best_cut_value = simulated_annealing_maxcut(G, initial_temp, final_temp, alpha, max_steps, init_partition)
output = []
for i in range(len(best_partition)):
    output.append(best_partition[i])
print("QAOA+SA result:", output)
print("Best cut value:", best_cut_value)
plot_result(graph, best_partition)
plt.show()
