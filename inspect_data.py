import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_graph(graph):
    # Get the initial and final node positions from the graph
    initial_pos = []
    final_pos = []
    for node_id in graph.nodes:
        node_props = graph.nodes[node_id]
        initial_pos.append(node_props['initial_position'])
        final_pos.append(node_props['final_position'])
        if node_props['contact_node']:
            contact_node_pos = node_props['initial_position']
    initial_pos = np.array(initial_pos)
    final_pos = np.array(final_pos)

    # Draw the force applied on the contact node
    contact_force = graph.graph['contact_force']

    # Draw all the edges
    initial_parent2child_quiver_list = []
    initial_child2parent_quiver_list = []
    final_parent2child_quiver_list = []
    final_child2parent_quiver_list = []
    for edge_idxs in graph.edges:
        edge_props = graph.edges[edge_idxs]
        if edge_props['parent2child']==1:
            initial_parent2child_quiver_list.append([graph.nodes[edge_idxs[0]]['initial_position'], edge_props['initial_edge_delta']])
            final_parent2child_quiver_list.append([graph.nodes[edge_idxs[0]]['final_position'], edge_props['final_edge_delta']])
        elif edge_props['parent2child']==-1:
            initial_child2parent_quiver_list.append([graph.nodes[edge_idxs[0]]['initial_position'], edge_props['initial_edge_delta']])
            final_child2parent_quiver_list.append([graph.nodes[edge_idxs[0]]['final_position'], edge_props['final_edge_delta']])
    initial_parent2child_quiver_list = np.array(initial_parent2child_quiver_list).reshape(-1, 6)
    initial_child2parent_quiver_list = np.array(initial_child2parent_quiver_list).reshape(-1, 6)
    final_parent2child_quiver_list = np.array(final_parent2child_quiver_list).reshape(-1, 6)
    final_child2parent_quiver_list = np.array(final_child2parent_quiver_list).reshape(-1, 6)

    # Plot the initial and final node positions in 3D space with equal axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot initial and final node positions
    ax.scatter(initial_pos[:,0], initial_pos[:,1], initial_pos[:,2], c='b', marker='o')
    ax.scatter(final_pos[:,0], final_pos[:,1], final_pos[:,2], c='r', marker='o')
    # Plot the contact force
    q0 = ax.quiver(contact_node_pos[0], contact_node_pos[1], contact_node_pos[2], contact_force[0], contact_force[1], contact_force[2], length=0.2, normalize=True, color='g')
    # Plot the edges
    for quiver_params in initial_parent2child_quiver_list:
        q1 = ax.quiver(quiver_params[0], quiver_params[1], quiver_params[2], quiver_params[3], quiver_params[4], quiver_params[5], color='r')
    for quiver_params in initial_child2parent_quiver_list:
        q2 = ax.quiver(quiver_params[0], quiver_params[1], quiver_params[2], quiver_params[3], quiver_params[4], quiver_params[5], color='b')
    for quiver_params in final_parent2child_quiver_list:
        q3 = ax.quiver(quiver_params[0], quiver_params[1], quiver_params[2], quiver_params[3], quiver_params[4], quiver_params[5], color='m')
    for quiver_params in final_child2parent_quiver_list:
        q4 = ax.quiver(quiver_params[0], quiver_params[1], quiver_params[2], quiver_params[3], quiver_params[4], quiver_params[5], color='c')
   

    # Set plot axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.0)

    # Set plot legend
    legend_labels = ['Normalized Contact Force', 'Initial Parent2Child Edges', 'Initial Child2Parent Edges', 'Final Parent2Child Edges', 'Final Child2Parent Edges']
    ax.legend([q0, q1, q2, q3, q4], legend_labels)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='data/graph_list.pkl', help='path to data pkl file')
    args = parser.parse_args()
    with open(args.data_path, 'rb') as f:
        graph_list = pickle.load(f)
    for g in graph_list:
        visualize_graph(g)
    