import cvxpy as cp
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pdra
from typing import List


# Distributed Quadratic Programming (QP), using accelerated gradient descent (AGD)
class NodeDQP(pdra.NodeDRABase):
    def __init__(self, q_i, g_i, iterations, gamma, a_i, **kwargs):
        self.q_i = q_i
        self.g_i = g_i

        super().__init__(iterations, gamma, pdra.AGD, a_i, **kwargs)

    @property
    def f_i(self) -> cp.Expression:
        return self.x_i @ self.q_i @ self.x_i / 2 + self.g_i @ self.x_i

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return []


# Collaborative Production, using subgradient method (SM)
class NodeCP(pdra.NodeDRABase):
    def __init__(self, c_profit_i, x_laboratory_i, iterations, gamma, a_material_i, **kwargs):
        self.c_profit_i = c_profit_i
        self.x_laboratory_i = x_laboratory_i

        # Set the solver to GLPK since it is more suitable for LP
        super().__init__(iterations, gamma, pdra.SM, a_material_i, solver=cp.GLPK, **kwargs)

    @property
    def f_i(self) -> cp.Expression:
        return -self.c_profit_i @ self.x_i

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return [self.x_i - self.x_laboratory_i <= 0]


def save_results(f_star, l_mat, r_vec, nodes, exper):
    err_series = sum([node.f_i_series for node in nodes.values()]) - f_star
    c_series = {f'{i}': node.c_i_series for i, node in nodes.items()}
    cons_series = sum([l_mat[i] @ node.x_i_series for i, node in nodes.items()]) - r_vec[:, np.newaxis]

    np.save(rf'..\data\{exper}\results\err_series.npy', err_series)
    np.savez(rf'..\data\{exper}\results\c_series.npz', **c_series)
    np.save(rf'..\data\{exper}\results\cons_series.npy', cons_series)


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Distributed Quadratic Programming
    # Collaborative Production
    experiment = 'Collaborative Production'

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
        T = 2000       # iteration number
        step_size = 3  # step size of the algorithm

        # Generate the model of the problem
        # di = 4
        # M = 3
        #
        # Q = {}
        # g = {}
        # A = {}
        #
        # for i in Nodes_set:
        #     Q_i_prime = np.random.randn(di, di)
        #     Q[i] = np.eye(di) + Q_i_prime @ Q_i_prime.T
        #     g[i] = np.random.randint(-5, 5, di)
        #     A[i] = np.random.randint(-50, 50, (M, di))
        #
        # np.savez(rf'..\data\{experiment}\model\Q.npz', **Q)
        # np.savez(rf'..\data\{experiment}\model\g.npz', **g)
        # np.savez(rf'..\data\{experiment}\model\A.npz', **A)
        #
        # vec = np.random.uniform(0, 1, 3)
        # pr = vec / vec.sum()
        # D = np.random.choice([1, 2, 3], 1000, p=pr)
        # np.save(rf'..\data\{experiment}\model\D.npy', D)

        Q = np.load(rf'..\data\{experiment}\model\Q.npz')
        g = np.load(rf'..\data\{experiment}\model\g.npz')
        A = np.load(rf'..\data\{experiment}\model\A.npz')

        D = np.load(rf'..\data\{experiment}\model\D.npy')

        proportion_of_1 = D[np.where(D == 1)].size / 1000
        proportion_of_2 = D[np.where(D == 2)].size / 1000
        proportion_of_3 = D[np.where(D == 3)].size / 1000

        # Vector 'b' represents the proportion of 1, 2, 3 in D respectively
        b = np.array([proportion_of_1, proportion_of_2, proportion_of_3])

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(A[i].shape[1]) for i in Nodes_set}

        cost = cp.sum([x[i] @ Q[i] @ x[i] / 2 + g[i] @ x[i] for i in Nodes_set])
        constraints = [cp.sum([A[i] @ x[i] for i in Nodes_set]) - b <= 0]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        F_star = problem.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        b_bar = pdra.resource_perturbation(epsilon, delta, Delta, b)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Initialize nodes and only send resources to node 1
        Nodes = {i: NodeDQP(Q[i], g[i], T, step_size, A[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeDQP(Q['1'], g['1'], T, step_size, A['1'], b_i=b_bar)

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
        save_results(F_star, A, b, Nodes, experiment)

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

        # Generate the model of the problem
        # di = 2
        # M = 2
        #
        # c_profit = {}
        # A_material = {}
        # x_laboratory = {}
        #
        # for i in Nodes_set:
        #     c_profit[i] = np.random.uniform(1, 2, di)
        #     A_material[i] = np.random.uniform(1, 5, (M, di))
        #     x_laboratory[i] = np.random.uniform(1, 5, di)
        #
        # np.savez(rf'..\data\Collaborative Production\model\c_profit.npz', **c_profit)
        # np.savez(rf'..\data\Collaborative Production\model\A_material.npz', **A_material)
        # np.savez(rf'..\data\Collaborative Production\model\x_laboratory.npz', **x_laboratory)
        #
        # b_material = np.random.uniform(18, 22, M)
        # np.save(rf'..\data\Collaborative Production\model\b_material.npy', b_material)

        c_profit = np.load(rf'..\data\{experiment}\model\c_profit.npz')
        A_material = np.load(rf'..\data\{experiment}\model\A_material.npz')
        x_laboratory = np.load(rf'..\data\{experiment}\model\x_laboratory.npz')

        b_material = np.load(rf'..\data\{experiment}\model\b_material.npy')

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(A_material[i].shape[1]) for i in Nodes_set}

        cost = cp.sum([-c_profit[i] @ x[i] for i in Nodes_set])

        laboratory_constraints = [x[i] - x_laboratory[i] <= 0 for i in Nodes_set]
        material_constraints = [cp.sum([A_material[i] @ x[i] for i in Nodes_set]) - b_material <= 0]

        constraints = material_constraints + laboratory_constraints

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.GLPK)

        F_star = problem.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.1

        b_material_bar = pdra.resource_perturbation(epsilon, delta, Delta, b_material)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Nodes = {i: NodeCP(c_profit[i], x_laboratory[i], T, step_size, A_material[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeCP(c_profit['1'], x_laboratory['1'], T, step_size, A_material['1'], b_i=b_material_bar)

        Edges = {e: (pdra.Edge(Nodes[e[0]], Nodes[e[1]]),
                     pdra.Edge(Nodes[e[1]], Nodes[e[0]]))
                 for e in Edges_set}

        for Node in Nodes.values():
            Node.start()

        for Node in Nodes.values():
            Node.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_results(F_star, A_material, b_material, Nodes, experiment)
