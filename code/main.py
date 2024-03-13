import cvxpy as cp
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pdra
from typing import List


# Distributed Quadratic Programming (QP), using accelerated gradient (AG) method
class NodeDQP(pdra.NodeAG):
    def __init__(self, iterations, dimension, gamma, f_i, a_i, b_i=None):
        super().__init__(iterations, dimension, gamma, f_i, a_i, b_i)

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return []

    @property
    def solver(self) -> str:
        return cp.OSQP


# Collaborative Production, using subgradient (SG) method
class NodeCP(pdra.NodeSG):
    def __init__(self, iterations, dimension, gamma, f_i, a_i, x_upper_i, b_i=None):
        self.x_upper_i = x_upper_i

        super().__init__(iterations, dimension, gamma, f_i, a_i, b_i)

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return [self.x_i - self.x_upper_i <= 0]

    @property
    def solver(self) -> str:
        return cp.GLPK


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Distributed Quadratic Programming
    # Collaborative Production
    experiment = 'Distributed Quadratic Programming'

    if experiment == 'Distributed Quadratic Programming':
        # Communication graph - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Number of nodes
        N = 9

        # The (name) sets of nodes and edges
        Nodes_set = {f'{i}' for i in range(1, N + 1)}
        Edges_set = {('1', '2'), ('2', '3'), ('3', '4'), ('3', '6'), ('3', '7'), ('4', '5'), ('6', '8'), ('7', '9')}

        # The coordinate of each node
        Nodes_pos = {'1': (-2, 0.6), '2': (-1, 0.3), '3': (0, 0), '4': (-1, -0.3), '5': (-2, -0.6), '6': (1, 0.3),
                     '7': (1, -0.3), '8': (2, 0.6), '9': (2, -0.6)}

        # Plot the graph
        G = nx.Graph()
        G.add_nodes_from(Nodes_set)
        G.add_edges_from(Edges_set)

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        nx.draw(G, pos=Nodes_pos, with_labels=True, ax=ax)

        plt.show()

        L = nx.laplacian_matrix(G).toarray()  # Laplacian matrix
        print(L)

        # plt.savefig(r'..\manuscript\src\figures\fig1.png', dpi=300, bbox_inches='tight')

        # Parameters initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 2000       # iteration number
        dim = 4        # dimension for the decision variable
        M = 3          # constraints number
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

        f = {i: lambda var, index=i: var @ Q[index] @ var / 2 + g[index] @ var for i in Nodes_set}

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(dim) for i in Nodes_set}

        centralized_cost = cp.sum([f[i](x[i]) for i in Nodes_set])
        centralized_constraints = [cp.sum([A[i] @ x[i] for i in Nodes_set]) - b <= 0]

        centralized_problem = cp.Problem(cp.Minimize(centralized_cost), centralized_constraints)
        centralized_problem.solve(solver=cp.OSQP)

        F_star = centralized_problem.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        b_bar = pdra.resource_perturbation(epsilon, delta, Delta, M, b)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Initialize nodes and only send resources to node 1
        Nodes = {i: NodeDQP(T, dim, step_size, f[i], A[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeDQP(T, dim, step_size, f['1'], A['1'], b_bar)

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
        pdra.save_data(F_star, Nodes, A, b, r'..\data\\' + experiment)

    elif experiment == 'Collaborative Production':
        # Communication graph - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        N = 15
        Nodes_set = {f'{i}' for i in range(1, N + 1)}
        Edges_set = {('1', '9'), ('1', '11'), ('2', '3'), ('2', '6'), ('2', '7'), ('2', '10'), ('6', '12'),
                     ('3', '15'), ('4', '8'), ('5', '14'), ('8', '15'), ('11', '13'), ('11', '14'), ('13', '15')}
        Nodes_pos = {'1': [0.13436424411240122, 0.8474337369372327], '2': [0.763774618976614, 0.2550690257394217],
                     '3': [0.59543508709194095, 0.4494910647887381], '4': [0.651592972722763, 0.7887233511355132],
                     '5': [0.0938595867742349, 0.02834747652200631], '6': [0.8357651039198697, 0.43276706790505337],
                     '7': [0.762280082457942, 0.0021060533511106927], '8': [0.4453871940548014, 0.7215400323407826],
                     '9': [0.22876222127045265, 0.9452706955539223], '10': [0.9014274576114836, 0.030589983033553536],
                     '11': [0.0254458609934608, 0.5414124727934966], '12': [0.9391491627785106, 0.38120423768821243],
                     '13': [0.21659939713061338, 0.4221165755827173], '14': [0.029040787574867943, 0.22169166627303505],
                     '15': [0.43788759365057206, 0.49581224138185065]}

        G = nx.Graph()
        G.add_nodes_from(Nodes_set)
        G.add_edges_from(Edges_set)

        fig1, ax1 = plt.subplots(1, 1)
        nx.draw(G, pos=Nodes_pos, with_labels=True, ax=ax1)

        plt.show()

        L = nx.laplacian_matrix(G).toarray()  # Laplacian matrix
        print(L)

        # plt.savefig(r'..\manuscript\src\figures\fig2.png', dpi=300, bbox_inches='tight')

        # Parameters initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 3000
        dim = 2
        M = 2
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
            x_laboratory[i] = pd.read_excel(r'..\data\\' + experiment + r'\node' + i + r'\x_lab.xlsx').values.reshape(-1)

        b_material = pd.read_excel(r'..\data\\' + experiment + r'\b_mat.xlsx').values.reshape(-1)

        f = {i: lambda var, i=i: -c_profit[i] @ var for i in Nodes_set}

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(dim) for i in Nodes_set}

        centralized_cost = cp.sum([f[i](x[i]) for i in Nodes_set])

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

        b_material_bar = pdra.resource_perturbation(epsilon, delta, Delta, M, b_material)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Nodes = {i: NodeCP(T, dim, step_size, f[i], A_material[i], x_laboratory[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeCP(T, dim, step_size, f['1'], A_material['1'], x_laboratory['1'], b_material_bar)

        Edges = {e: (pdra.Edge(Nodes[e[0]], Nodes[e[1]]),
                     pdra.Edge(Nodes[e[1]], Nodes[e[0]]))
                 for e in Edges_set}

        for Node in Nodes.values():
            Node.start()

        for Node in Nodes.values():
            Node.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        pdra.save_data(F_star, Nodes, A_material, b_material, r'..\data\\' + experiment)
