from lib import MyGraph, generate_graph
goal = 30
start = 0
point_set = generate_graph(goal, p=0.4, p2=0.03)
print(point_set)
graph = MyGraph(goal+1, point_set, start, goal)
graph.draw_graph()
graph.train_until_convergence()
graph.draw_graph_path_solution()