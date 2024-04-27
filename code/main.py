import cvxpy as cp
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pdra
from typing import List


# Distributed Quadratic Programming (QP), using accelerated gradient (AG) method
class NodeDQP(pdra.NodeAG):
    def __init__(self, q_i, g_i, iterations, gamma, a_i, b_i=None):
        self.q_i = q_i
        self.g_i = g_i

        super().__init__(iterations, gamma, a_i, b_i)

    @property
    def f_i(self) -> cp.Expression:
        return self.x_i @ self.q_i @ self.x_i / 2 + self.g_i @ self.x_i

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return []


# Collaborative Production, using subgradient (SG) method
class NodeCP(pdra.NodeSG):
    def __init__(self, c_profit_i, x_laboratory_i, iterations, gamma, a_material_i, b_material_i=None):
        self.c_profit_i = c_profit_i
        self.x_laboratory_i = x_laboratory_i

        super().__init__(iterations, gamma, a_material_i, b_material_i)

        # Set the solver to GLPK for it is more suitable for LP
        self.solver = cp.GLPK

    @property
    def f_i(self) -> cp.Expression:
        return -self.c_profit_i @ self.x_i

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return [self.x_i - self.x_laboratory_i <= 0]


def save_results(f_star, nodes, directory_path) -> None:
    # Error between F_iter and F_star
    f_iter = sum([node.f_i_iter for node in nodes.values()])

    err = f_iter - f_star
    df = pd.DataFrame(err)
    df.to_excel(directory_path + r'\err.xlsx', index=False)

    # Lagrange multipliers
    for i, node in nodes.items():
        df = pd.DataFrame(node.c_i_iter)
        df.to_excel(directory_path + r'\node' + i + r'\c_iter.xlsx', index=False)

    # The iterations of coupling constraints
    cons_iter = sum([node.a_i @ node.x_i_iter for i, node in nodes.items()]) - nodes['1'].b_i[:, np.newaxis]

    df = pd.DataFrame(cons_iter)
    df.to_excel(directory_path + r'\cons_iter.xlsx', index=False)


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Distributed Quadratic Programming
    # Collaborative Production
    experiment = 'Distributed Quadratic Programming'

    # Some options for the figure of the communication graph
    options = {'with_labels': True,
               'font_size': 20,
               'node_color': 'white',
               'node_size': 1000,
               'edgecolors': 'black',
               'linewidths': 1.5,
               'width': 1.5}

    if experiment == 'Distributed Quadratic Programming':
        # Communication graph - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Number of nodes
        N = 9

        # The (name) sets of nodes and edges
        Nodes_set = {f'{i}' for i in range(1, N + 1)}
        Edges_set = {('1', '2'), ('2', '3'), ('3', '4'), ('3', '6'), ('3', '7'), ('4', '5'), ('6', '8'), ('7', '9')}

        # The coordinate of each node
        Nodes_pos = {'1': (-2, 1), '2': (-1, 0.5), '3': (0, 0), '4': (-1, -0.5), '5': (-2, -1), '6': (1, 0.5),
                     '7': (1, -0.5), '8': (2, 1), '9': (2, -1)}

        # Plot the graph
        G = nx.Graph()
        G.add_nodes_from(Nodes_set)
        G.add_edges_from(Edges_set)

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.set_ylim([-1.5, 1.5])

        nx.draw(G, pos=Nodes_pos, ax=ax, **options)

        plt.show()

        L = nx.laplacian_matrix(G).toarray()  # Laplacian matrix
        print(L)

        # fig.savefig(r'..\manuscript\src\figures\fig1.png', dpi=300, bbox_inches='tight')

        # Parameters initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 2000  # iteration number
        step_size = 3  # step size of the algorithm

        Q = {}
        g = {}
        A = {}

        # Generate the parameters of the problem
        # for i in Nodes_set:
        #     Q_i_prime = np.random.randn(di, di)
        #     Q[i] = np.eye(di) + Q_i_prime @ Q_i_prime.T
        #     df = pd.DataFrame(Q[i])
        #     df.to_excel(r'..\data\Distributed Quadratic Programming\node' + i + r'\Q.xlsx', index=False)
        #
        #     g[i] = np.random.randint(-5, 5, di)
        #     df = pd.DataFrame(g[i])
        #     df.to_excel(r'..\data\Distributed Quadratic Programming\node' + i + r'\g.xlsx', index=False)
        #
        #     A[i] = np.random.randint(-50, 50, (M, di))
        #     print(np.linalg.matrix_rank(A[i]))
        #     df = pd.DataFrame(A[i])
        #     df.to_excel(r'..\data\Distributed Quadratic Programming\node' + i + r'\A.xlsx', index=False)
        #
        # vec = np.random.uniform(0, 1, 3)
        # pr = vec / vec.sum()
        # D = np.random.choice([1, 2, 3], 1000, p=pr)
        # df = pd.DataFrame(D)
        # df.to_excel(r'..\data\Distributed Quadratic Programming\D.xlsx', index=False)

        for i in Nodes_set:
            Q[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\Q.xlsx').values
            g[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\g.xlsx').values.reshape(-1)
            A[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\A.xlsx').values

        D = pd.read_excel(r'..\data\\' + experiment + r'\D.xlsx').values.reshape(-1)
        proportion_of_1 = D[np.where(D == 1)].size / 1000
        proportion_of_2 = D[np.where(D == 2)].size / 1000
        proportion_of_3 = D[np.where(D == 3)].size / 1000

        # Vector 'b' represents the proportion of 1, 2, 3 in D respectively
        b = np.array([proportion_of_1, proportion_of_2, proportion_of_3])

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(A[i].shape[1]) for i in Nodes_set}

        centralized_cost = cp.sum([x[i] @ Q[i] @ x[i] / 2 + g[i] @ x[i] for i in Nodes_set])
        coupling_constraints = [cp.sum([A[i] @ x[i] for i in Nodes_set]) - b <= 0]

        centralized_problem = cp.Problem(cp.Minimize(centralized_cost), coupling_constraints)
        centralized_problem.solve(solver=cp.OSQP)

        F_star = centralized_problem.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        b_bar = pdra.resource_perturbation(epsilon, delta, Delta, b)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Initialize nodes and only send resources to node 1
        Nodes = {i: NodeDQP(Q[i], g[i], T, step_size, A[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeDQP(Q['1'], g['1'], T, step_size, A['1'], b_bar)

        # Build up communication edges
        Edges = {e[0] + '<->' + e[1]: (pdra.Edge(Nodes[e[0]], Nodes[e[1]]),
                                       pdra.Edge(Nodes[e[1]], Nodes[e[0]]))
                 for e in Edges_set}

        # Disconnect edge 1<->2
        # Edges['1<->2'][0].disconnect()
        # Edges['1<->2'][1].disconnect()

        for Node in Nodes.values():
            Node.start()

        for Node in Nodes.values():
            Node.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_results(F_star, Nodes, r'..\data\\' + experiment)

    elif experiment == 'Collaborative Production':
        # Communication graph - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        N = 15
        Nodes_set = {f'{i}' for i in range(1, N + 1)}
        Edges_set = {('1', '9'), ('1', '11'), ('2', '3'), ('2', '6'), ('2', '7'), ('2', '10'), ('6', '12'),
                     ('3', '15'), ('4', '8'), ('5', '14'), ('8', '15'), ('11', '13'), ('11', '14'), ('13', '15')}
        Nodes_pos = {'1': [-3, 1.5], '2': [2, 1],
                     '3': [1, 0.5], '4': [2, -1],
                     '5': [-4, 0], '6': [1, 1.5],
                     '7': [3, 0.5], '8': [1, -0.5],
                     '9': [-4, 2], '10': [3, 1.5],
                     '11': [-2, 1], '12': [0, 2],
                     '13': [-1, 0.5], '14': [-3, 0.5],
                     '15': [0, 0]}

        G = nx.Graph()
        G.add_nodes_from(Nodes_set)
        G.add_edges_from(Edges_set)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_aspect(1)

        nx.draw(G, pos=Nodes_pos, ax=ax, **options)

        plt.show()

        L = nx.laplacian_matrix(G).toarray()  # Laplacian matrix
        print(L)

        # fig.savefig(r'..\manuscript\src\figures\fig2.png', dpi=300, bbox_inches='tight')

        # Parameters initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 3000
        step_size = 10

        c_profit = {}
        A_material = {}
        x_laboratory = {}

        # Generate the parameters of the problem
        # for i in Nodes_set:
        #     c_pro[i] = np.random.uniform(1, 2, di)
        #     df = pd.DataFrame(c_pro[i])
        #     df.to_excel(r'..\data\Collaborative Production\node' + i + r'\c_pro.xlsx', index=False)
        #
        #     A_mat[i] = np.random.uniform(1, 5, (M, di))
        #     df = pd.DataFrame(A_mat[i])
        #     df.to_excel(r'..\data\Collaborative Production\node' + i + r'\a_mat.xlsx', index=False)
        #
        #     x_lab[i] = np.random.uniform(1, 5, di)
        #     df = pd.DataFrame(x_lab[i])
        #     df.to_excel(r'..\data\Collaborative Production\node' + i + r'\x_lab.xlsx', index=False)
        #
        # b_mat = np.random.uniform(18, 22, M)
        # df = pd.DataFrame(b_mat)
        # df.to_excel(r'..\data\Collaborative Production\b_mat.xlsx', index=False)

        for i in Nodes_set:
            c_profit[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\c_pro.xlsx').values.reshape(-1)
            A_material[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\a_mat.xlsx').values
            x_laboratory[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\x_lab.xlsx').values.reshape(
                -1)

        b_material = pd.read_excel(r'..\data\\' + experiment + r'\b_mat.xlsx').values.reshape(-1)

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(A_material[i].shape[1]) for i in Nodes_set}

        centralized_cost = cp.sum([-c_profit[i] @ x[i] for i in Nodes_set])

        laboratory_constraints = [x[i] - x_laboratory[i] <= 0 for i in Nodes_set]
        material_constraints = [cp.sum([A_material[i] @ x[i] for i in Nodes_set]) - b_material <= 0]

        centralized_constraints = material_constraints + laboratory_constraints

        centralized_problem = cp.Problem(cp.Minimize(centralized_cost), centralized_constraints)
        centralized_problem.solve(solver=cp.GLPK)

        F_star = centralized_problem.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.1

        b_material_bar = pdra.resource_perturbation(epsilon, delta, Delta, b_material)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Nodes = {i: NodeCP(c_profit[i], x_laboratory[i], T, step_size, A_material[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeCP(c_profit['1'], x_laboratory['1'], T, step_size, A_material['1'], b_material_bar)

        Edges = {e: (pdra.Edge(Nodes[e[0]], Nodes[e[1]]),
                     pdra.Edge(Nodes[e[1]], Nodes[e[0]]))
                 for e in Edges_set}

        for Node in Nodes.values():
            Node.start()

        for Node in Nodes.values():
            Node.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_results(F_star, Nodes, r'..\data\\' + experiment)
