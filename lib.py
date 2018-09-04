import numpy as np
import networkx as nx
import pylab as plt
import math

class MyGraph(object):

    def __init__(self, size_, point_set, start, goal, gamma=0.8):
        self.size = size_
        self.point_set = point_set
        self.nodes = [i for i in range(self.size)]
        print(self.nodes)
        self.goal = goal
        self.start = start
        self.state = start
        self.gamma = gamma
        self.__create_R_matrix()
        self.__create_Q_matrix()
        self.state = start
        self.path = []

    def __get_available_moves(self):
        row = self.R[self.state, ]
        available_moves = np.where(row >= 0)[1]
        while len(available_moves) == 0:
            self.__change_state()
            row = self.R[self.state,]
            available_moves = np.where(row >= 0)[1]
        return available_moves

    @staticmethod
    def sample_move(available_moves):
        choice = np.random.choice(available_moves, 1)[0]
        return choice

    def __create_R_matrix(self):
        self.R = np.matrix(np.ones(shape=(self.size, self.size))) * -1
        for point in self.point_set:
            start_p, end_p = point
            if end_p == self.goal:
                self.R[point] = 100
            else:
                self.R[point] = 0
            if start_p == self.goal:
                self.R[point[::-1]] = 100
            else:
                self.R[point[::-1]] = 0
        self.R[self.goal, self.goal] = 100

    def __change_state(self):
        self.state = np.random.randint(0, self.size)

    def __create_Q_matrix(self):
        self.Q = np.matrix(np.zeros(shape=(self.size, self.size)))


    def __update_Q_matrix(self, move):
        current_state = self.state
        max_index = np.where(self.Q[move, ] == np.max(self.Q[move, ]))[1]
        if max_index.shape[0] > 1:
            max_index = np.random.choice(max_index, 1)[0]
        else:
            max_index = max_index[0]
        max_value = self.Q[move, max_index]
        self.Q[current_state, move] = self.R[current_state, move] + max_value * self.gamma
        print('max_value', self.R[current_state, move] + self.gamma * max_value)

        if np.max(self.Q) > 0:
            return np.sum(self.Q/np.max(self.Q) * 100)
        else:
            return 0

    def train_until_convergence(self, max_iter = 10000):
        scores = []
        for i in range(0, max_iter):
            moves = self.__get_available_moves()
            move = MyGraph.sample_move(moves)
            score = self.__update_Q_matrix(move)
            print(score)
            scores.append(score)
            self.__change_state()
        max_value = np.max(self.Q)
        for i in range(self.size):
            for j in range(self.size):
                self.Q[i, j] = self.Q[i, j]/max_value * 100
        self.get_path()
        return scores

    def get_path(self):
        curr_state = self.start
        curr_edge = [self.start]
        while curr_state != self.goal:
            next_step = np.where(self.Q[curr_state,] == np.max(self.Q[curr_state]))[1]
            if len(next_step) > 1:
                next_step = np.random.choice(next_step, 1)[0]
                curr_edge.append(next_step)
            else:
                next_step = next_step[0]
                curr_edge.append(next_step)
            self.path.append(tuple(curr_edge))
            curr_state = next_step
            curr_edge = [curr_state, ]
        print(self.path)
        return self.path

    def draw_graph(self):
        self.fig = plt.figure(figsize=(12, 7))
        ax1 = self.fig.add_subplot(121)
        my_graph = nx.Graph()
        my_graph.add_edges_from(self.point_set, color="black", weight=1)
        self.layout = nx.spring_layout(my_graph, k=10/math.sqrt(my_graph.order()))
        nx.draw_networkx_labels(my_graph, self.layout)
        nx.draw_networkx_edges(my_graph, self.layout)
        colors = []
        for node in my_graph:
            if self.start<node<self.goal:
                colors.append('r')
            else:
                colors.append('g')
        nx.draw_networkx_nodes(my_graph, self.layout, node_color=colors)

    def draw_graph_path_solution(self):
        my_graph = nx.Graph()
        my_graph.add_edges_from(self.point_set, color="black", weight=1)
        for e1, e2 in self.path:
            my_graph[e1][e2]['color'] = "blue"
            my_graph[e1][e2]["weight"] = 3

        edges = my_graph.edges()
        colors_ = [my_graph[a][b]['color'] for a, b in edges]
        weights = [my_graph[u][v]['weight'] for u, v in edges]
        colors = []
        for node in my_graph:
            if self.start < node < self.goal:
                colors.append('r')
            else:
                colors.append('g')
        self.fig.add_subplot(122)
        nx.draw(my_graph, self.layout, edges=edges, edge_color=colors_, node_color=colors, width=weights)
        nx.draw_networkx_labels(my_graph, self.layout)
        plt.savefig("result.png")


def generate_graph(end_node, p=0.2, p2=0.3):
    nodes = [i for i in range(0, end_node+1)]
    nodes_for_path = [val for val in nodes][1:-1]
    path = []
    curr_node = [0, ]
    for n in nodes_for_path:
        if np.random.random()<=p:
            curr_node.append(n)
            path.append(tuple(curr_node))
            curr_node = [n,]
    curr_node += [end_node]
    path.append(tuple(curr_node))
    other_paths = []
    for n1 in nodes:
        curr_node = [n1, ]
        for n2 in nodes:
            if n2 != n1 and np.random.random() <= p2:
                curr_node.append(n2)
                other_paths.append(tuple(curr_node))
                curr_node = [n1, ]
    return path+other_paths


# example graph
if __name__ == '__main__':
    point_set = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (5, 1), (2, 3), (3, 6), (6, 7), (7, 1), (7, 5), (8, 3), (6, 9)]

    goal = 9
    graph = MyGraph(size_=10, point_set=point_set, start=0, goal=9)
    graph.draw_graph()
    graph.train_until_convergence()
    path = graph.get_path()
    graph.draw_graph_path_solution()