import cvxpy as cp
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import dissys
import trunclap as tl


# The common parts of the two modes of our algorithm
class NodePDRABase(dissys.Node):
    def __init__(self, iterations, dimension, gamma, f_i, a_i: np.ndarray, b_i):
        super().__init__()
        # Iteration number
        self.T = iterations

        # Local decision variables
        self.x_i = cp.Variable(dimension)
        # Array for storing the iterations of the local decision variables
        self.x_iter = np.zeros((dimension, self.T))

        # Step size for accelerated gradient descent or the step length for subgradient method
        self.gamma = gamma

        # Local objective function and constraint coefficient matrix
        self.f_i = f_i
        self.a_i = a_i

        # Array for storing the iterations of local objective function
        self.f_iter = np.zeros(self.T)

        # The number of local constraints
        self.cons_num = a_i.shape[0]

        # Only node 1 can receive resource, other nodes set it to 0 by default
        self.b_i = b_i if b_i is not None else np.zeros(self.cons_num)

        # Auxiliary variables y and Lagrange multipliers c
        self.y = np.zeros(self.cons_num)
        self.c = np.zeros(self.cons_num)
        # Arrays for storing the iterations of auxiliary variables y and Lagrange multipliers c
        self.y_iter = np.zeros((self.cons_num, self.T))
        self.c_iter = np.zeros((self.cons_num, self.T))

        # l_i @ y, where l_i is the ith row of laplacian matrix L,
        # it is set as a Parameter class in cvxpy for it changes during every iteration
        self.li_y = cp.Parameter(self.cons_num)


# Experiment 1
class NodeAGD(NodePDRABase):
    def __init__(self, iterations, dimension, gamma, f_i, a_i, b_i=None):
        super().__init__(iterations, dimension, gamma, f_i, a_i, b_i)

        # Auxiliary variables in accelerated gradient method
        self.w = np.zeros(self.cons_num)
        self.theta = 1

        # Model the local problem
        cons = [self.a_i @ self.x_i + self.li_y - self.b_i <= 0]
        self.prob = cp.Problem(cp.Minimize(self.f_i(self.x_i)), cons)

    def run(self) -> None:
        for k in range(self.T):
            # Exchange the information of y with neighbors
            self.send_to_neighbors(self.y)
            y_j_lst = self.get_from_neighbors()

            # Update the parameters of the local problem
            self.li_y.value = self.y * self.in_degree - sum(y_j_lst)

            self.prob.solve(solver=cp.OSQP)

            # Update the Lagrange multipliers
            self.c = self.prob.constraints[0].dual_value

            # Store the required data
            self.y_iter[:, k] = self.y
            self.c_iter[:, k] = self.c
            self.x_iter[:, k] = self.x_i.value
            self.f_iter[k] = self.prob.value

            # Exchange the information of c with neighbors
            self.send_to_neighbors(self.c)
            c_j_lst = self.get_from_neighbors()

            # Calculate the gradient
            li_c = self.c * self.in_degree - sum(c_j_lst)

            # Accelerated gradient method
            w_temp = self.w
            theta_temp = self.theta

            self.w = self.y - self.gamma * li_c
            self.theta = (1 + np.sqrt(1 + 4 * (self.theta ** 2))) / 2
            self.y = self.w + ((theta_temp - 1) / self.theta) * (self.w - w_temp)


# Experiment 2
class NodeSG(NodePDRABase):
    def __init__(self, iterations, dimension, gamma, f_i, a_i, x_upper_i, b_i=None):
        super().__init__(iterations, dimension, gamma, f_i, a_i, b_i)

        # Model the local problem
        cons = [self.a_i @ self.x_i + self.li_y - self.b_i <= 0,
                self.x_i - x_upper_i <= 0]
        self.prob = cp.Problem(cp.Minimize(self.f_i(self.x_i)), cons)

    def run(self) -> None:
        for k in range(self.T):
            # Exchange the information of y with neighbors
            self.send_to_neighbors(self.y)
            y_j_lst = self.get_from_neighbors()

            # Update the parameters of the local problem
            self.li_y.value = self.y * self.in_degree - sum(y_j_lst)

            self.prob.solve(solver=cp.GLPK)

            # Update the Lagrange multipliers
            self.c = self.prob.constraints[0].dual_value

            # Store the required data
            self.x_iter[:, k] = self.x_i.value
            self.f_iter[k] = self.prob.value
            self.y_iter[:, k] = self.y
            self.c_iter[:, k] = self.c

            # Exchange the information of c with neighbors
            self.send_to_neighbors(self.c)
            c_j_lst = self.get_from_neighbors()

            # Calculate the subgradient
            li_c = self.c * self.in_degree - sum(c_j_lst)

            # Subgradient method
            self.y = self.y - (self.gamma / np.sqrt(k + 1)) * li_c


def resource_perturbation(eps, delt, sens, rsrc_dim, rsrc):
    s = (sens / eps) * np.log(rsrc_dim * (np.exp(eps) - 1) / delt + 1)
    trunc_lap = tl.TruncatedLaplace(-s, s, 0, sens / eps)
    rsrc_perturbed = rsrc - s * np.ones(rsrc_dim) + trunc_lap(rsrc_dim)

    return rsrc_perturbed


def save_data(nodes, f_star, a_dic, b_src, exp):
    # Error between F_iter and F_star
    f_iter = sum([node.f_iter for node in nodes.values()])

    err = f_iter - f_star
    df = pd.DataFrame(err)
    df.to_excel(r'..\data\experiment' + exp + r'\err.xlsx', index=False)

    # Lagrange multipliers
    for i, node in nodes.items():
        df = pd.DataFrame(node.c_iter)
        df.to_excel(r'..\data\experiment' + exp + r'\node' + i + r'\c_iter.xlsx', index=False)

    # The iterations of coupling constraints
    cons_iter = sum([a_dic[i] @ node.x_iter for i, node in nodes.items()]) - b_src[:, np.newaxis]

    df = pd.DataFrame(cons_iter)
    df.to_excel(r'..\data\experiment' + exp + r'\cons_iter.xlsx', index=False)


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    experiment = '1'

    if experiment == '1':
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

        # Generate matrices
        # for i in Nodes_set:
        #     Q_i_prime = np.random.randn(di, di)
        #     Q[i] = np.eye(di) + Q_i_prime @ Q_i_prime.T
        #     df = pd.DataFrame(Q[i])
        #     df.to_excel(r'..\data\experiment1\node' + i + r'\Q.xlsx', index=False)
        #
        #     g[i] = np.random.randint(-5, 5, di)
        #     df = pd.DataFrame(g[i])
        #     df.to_excel(r'..\data\experiment1\node' + i + r'\g.xlsx', index=False)
        #
        #     A[i] = np.random.randint(-50, 50, (M, di))
        #     print(np.linalg.matrix_rank(A[i]))
        #     df = pd.DataFrame(A[i])
        #     df.to_excel(r'..\data\experiment1\node' + i + r'\A.xlsx', index=False)
        #
        # vec = np.random.uniform(0, 1, 3)
        # pr = vec / vec.sum()
        # D = np.random.choice([1, 2, 3], 1000, p=pr)
        # df = pd.DataFrame(D)
        # df.to_excel(r'..\data\experiment1\D.xlsx', index=False)

        for i in Nodes_set:
            Q[i] = pd.read_excel(r'..\data\experiment1\node' + i + r'\Q.xlsx').values
            g[i] = pd.read_excel(r'..\data\experiment1\node' + i + r'\g.xlsx').values.reshape(-1)
            A[i] = pd.read_excel(r'..\data\experiment1\node' + i + r'\A.xlsx').values

        D = pd.read_excel(r'..\data\experiment1\D.xlsx').values.reshape(-1)
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

        b_bar = resource_perturbation(epsilon, delta, Delta, M, b)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Initialize nodes and only send resources to node 1
        Nodes = {i: NodeAGD(T, dim, step_size, f[i], A[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeAGD(T, dim, step_size, f['1'], A['1'], b_bar)

        # Build up communication edges, all edges weight 1
        Edges = {e: (dissys.Edge(Nodes[e[0]], Nodes[e[1]], 1),
                     dissys.Edge(Nodes[e[1]], Nodes[e[0]], 1))
                 for e in Edges_set}

        # Disconnect edge 1 <-> 2
        # Edges[('1', '2')][0].disconnect()
        # Edges[('1', '2')][1].disconnect()

        for Node in Nodes.values():
            Node.start()

        for Node in Nodes.values():
            Node.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_data(Nodes, F_star, A, b, experiment)

    elif experiment == '2':
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

        # Generate matrices
        # for i in Nodes_set:
        #     c_pro[i] = np.random.uniform(1, 2, di)
        #     df = pd.DataFrame(c_pro[i])
        #     df.to_excel(r'..\data\experiment2\node' + i + r'\c_pro.xlsx', index=False)
        #
        #     A_mat[i] = np.random.uniform(1, 5, (M, di))
        #     df = pd.DataFrame(A_mat[i])
        #     df.to_excel(r'..\data\experiment2\node' + i + r'\a_mat.xlsx', index=False)
        #
        #     x_lab[i] = np.random.uniform(1, 5, di)
        #     df = pd.DataFrame(x_lab[i])
        #     df.to_excel(r'..\data\experiment2\node' + i + r'\x_lab.xlsx', index=False)
        #
        # b_mat = np.random.uniform(18, 22, M)
        # df = pd.DataFrame(b_mat)
        # df.to_excel(r'..\data\experiment2\b_mat.xlsx', index=False)

        for i in Nodes_set:
            c_profit[i] = pd.read_excel(r'..\data\experiment2\node' + i + r'\c_pro.xlsx').values.reshape(-1)
            A_material[i] = pd.read_excel(r'..\data\experiment2\node' + i + r'\a_mat.xlsx').values
            x_laboratory[i] = pd.read_excel(r'..\data\experiment2\node' + i + r'\x_lab.xlsx').values.reshape(-1)

        b_material = pd.read_excel(r'..\data\experiment2\b_mat.xlsx').values.reshape(-1)

        f = {i: lambda var, i=i: -c_profit[i] @ var for i in Nodes_set}

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {i: cp.Variable(dim) for i in Nodes_set}

        centralized_cost = cp.sum([f[i](x[i]) for i in Nodes_set])

        material_constraints = [cp.sum([A_material[i] @ x[i] for i in Nodes_set]) - b_material <= 0]
        laboratory_constraints = [x[i] - x_laboratory[i] <= 0 for i in Nodes_set]

        centralized_constraints = material_constraints + laboratory_constraints

        centralized_problem = cp.Problem(cp.Minimize(centralized_cost), centralized_constraints)
        centralized_problem.solve(solver=cp.GLPK)

        F_star = centralized_problem.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.1

        b_material_bar = resource_perturbation(epsilon, delta, Delta, M, b_material)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Initialize nodes and only send the perturbed resources to node 1
        Nodes = {i: NodeSG(T, dim, step_size, f[i], A_material[i], x_laboratory[i]) for i in Nodes_set if i != '1'}
        Nodes['1'] = NodeSG(T, dim, step_size, f['1'], A_material['1'], x_laboratory['1'], b_material_bar)

        # Build up communication edges
        Edges = {e: (dissys.Edge(Nodes[e[0]], Nodes[e[1]], 1),
                     dissys.Edge(Nodes[e[1]], Nodes[e[0]], 1))
                 for e in Edges_set}

        for Node in Nodes.values():
            Node.start()

        for Node in Nodes.values():
            Node.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_data(Nodes, F_star, A_material, b_material, experiment)
