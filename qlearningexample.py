import numpy as np
import pylab as plt

point_set = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (5, 1), (2, 3), (3, 6), (6, 7), (7, 1), (7, 5), (8, 3), (6, 9)]

goal = 9


# Constructing the R matrix
R = np.matrix(np.ones(shape=(10, 10)))*-1
#print(R)


for point in point_set:
    start_p, end_p = point
    if end_p == goal:
        print("yos")
        R[point] = 100
    else:
        R[point] = 0
    if start_p == goal:
        R[point[::-1]] = 100
    else:
        R[point[::-1]] = 0

R[goal, goal] = 100

# Constructing the Q matrix
Q = np.matrix(np.zeros(shape=(10, 10)))
gamma = 0.8
initial_state = 5

def get_avaiable_moves(state):
    row = R[state, ]
    avaiable_moves = np.where(row >= 0)[1]
    return avaiable_moves

def sample_move(avaiable_moves):
    choice = np.random.choice(avaiable_moves, 1)[0]
    return choice

def update_Q_matrix(current_state, move, gamma=0.8):
    max_index = np.where(Q[move, ] == np.max(Q[move, ]))[1]
    if max_index.shape[0] > 1:
        max_index = np.random.choice(max_index, 1)[0]
    else:
        max_index = max_index[0]
    max_value = Q[move, max_index]
    Q[current_state, move] = R[current_state, move] + max_value * gamma
    print('max_value', R[current_state, move] + gamma * max_value)

    if np.max(Q) > 0:
        return np.sum(Q/np.max(Q) * 100)
    else:
        return 0

scores = []
for i in range(0, 5000):
    curr_state = np.random.randint(0, 10)
    moves = get_avaiable_moves(curr_state)
    move = sample_move(moves)
    score = update_Q_matrix(curr_state, move)
    print(score)
    scores.append(score)

print("Trained Q matrix:")
print(Q/np.max(Q)*100)
'''
plt.plot(scores)
plt.show()
'''

path = []
curr_state = 7
curr_edge = [7]
while curr_state != 9:
    next_step = np.where(Q[curr_state,] == np.max(Q[curr_state]))[1]
    if len(next_step) > 1:
        next_step = np.random.choice(next_step, 1)[0]
        curr_edge.append(next_step)
    else:
        next_step = next_step[0]
        curr_edge.append(next_step)
    path.append(tuple(curr_edge))
    curr_state = next_step
    curr_edge = [curr_state, ]
print(path)

# To draw the graph

import networkx as nx
my_graph = nx.Graph()
my_graph.add_edges_from(point_set, color="black", weight=1)
for e1, e2 in path:
    my_graph[e1][e2]['color']="blue"
    my_graph[e1][e2]["weight"] = 3
layout = nx.spring_layout(my_graph)
edges = my_graph.edges()
colors = [my_graph[a][b]['color'] for a,b in edges]
weights = [my_graph[u][v]['weight'] for u,v in edges]
nx.draw(my_graph, layout, edges=edges, edge_color=colors, width=weights)
nx.draw_networkx_labels(my_graph, layout)

plt.show()