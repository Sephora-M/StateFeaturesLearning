import networkx as nx
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt


def creat_graph(m1, m2):
    m_1 = 2*m1+m2
    m_2 = m1+m2
    g = nx.barbell_graph(m1, m2)

    color = ['b','g','r','y']
    color_1 = color[0]
    color_2 = color[1]
    color_3 = color[2]
    node_colors = []
    
    for i in range(g.number_of_nodes()):
        if i < m1-1:
            node_colors.append(color_1)
        else:
            if i == m1 - 1:
                node_colors.append(color_2)
            else:
                if i < m_2:
                    node_colors.append(color_3)
                else:
                    if i == m_2:
                        node_colors.append(color_2)
                    else:
                        node_colors.append(color_1)
#    g.add_node(m+1)
#    g.add_node(m+2)
#    g.add_node(m+3)
#    g.add_node(m+4)
#    g.add_edge(7,m+1)
#    g.add_edge(m+1,m+2)
#    g.add_edge(m+2,m+3)
#    g.add_edge(m+2,m+4)
    adj = nx.adjacency_matrix(g)
    features = np.identity(adj.shape[0])
    nx.draw(g, with_labels = True, node_color = node_colors)
    print(adj.todense())
    plt.show()
    
    return adj, features, node_colors


def creat_club_graph():
    G = nx.karate_club_graph()
    nx.draw(G, with_labels=True)
    adj = nx.adjacency_matrix(G)
    features = np.identity(adj.shape[0])
    plt.show()
    return adj, features


def read_graph(edge_list):
    '''
    Reads the input network in networkx.
    '''

    G = nx.read_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    G = G.to_undirected()

    adj = nx.adjacency_matrix(G)
    features = np.identity(adj.shape[0])

    #nx.draw_spectral(G, with_labels=True, node_color=node_colors)
    return adj, features