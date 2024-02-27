import cvxpy as cvx
import numpy as np
import pandas as pd
import dissys
import trunclap as tl


class NodeExp1(dissys.Node):
    def __init__(self, name, edges_send, edges_recv, iterations, dim, gamma, f_i, a_i: np.ndarray, b_i=None):
        super().__init__(name, edges_send, edges_recv)
        # Iteration numbers
        self.T = iterations

        # Local decision variables
        self.x_i = cvx.Variable(dim)
        # Array for storing the iterations of the local decision variables
        self.x_iter = np.zeros((dim, self.T))

        # Step size for accelerated gradient method
        self.gamma = gamma

        # Local objective function and constraint coefficient matrix
        self.f_i = f_i
        self.a_i = a_i
        # The number of local constraints
        self.cons_num = a_i.shape[0]

        # Auxiliary variables y and Lagrange multipliers c
        self.y = np.zeros(self.cons_num)
        self.c = np.zeros(self.cons_num)
        # Arrays for storing the iterations of auxiliary variables y and Lagrange multipliers c
        self.y_iter = np.zeros((self.cons_num, self.T))
        self.c_iter = np.zeros((self.cons_num, self.T))

        # Auxiliary variables in accelerated gradient method
        self.w = np.zeros(self.cons_num)
        self.theta = 1

        # Array for storing the iterations of local objective function
        self.f_iter = np.zeros(self.T)

        # Model the local problem
        self.li_y = cvx.Parameter(self.cons_num)
        cons = [
            cvx.constraints.NonPos(self.a_i @ self.x_i + self.li_y - (np.zeros(self.cons_num) if b_i is None else b_i))]
        self.prob = cvx.Problem(cvx.Minimize(self.f_i(self.x_i)), cons)

    def run(self) -> None:
        for k in range(self.T):
            # Exchange the information of y with neighbors
            self.send_to_neighbors(self.y)
            y_j_lst = self.recv_from_neighbors()

            # Update the parameters of the local problem
            self.li_y.value = self.y * self.in_degree - sum(y_j_lst)

            self.prob.solve(solver=cvx.OSQP)

            # Update the Lagrange multipliers
            self.c = self.prob.constraints[0].dual_value

            # Store the required data
            self.y_iter[:, k] = self.y
            self.c_iter[:, k] = self.c
            self.x_iter[:, k] = self.x_i.value
            self.f_iter[k] = self.prob.value

            # Exchange the information of c with neighbors
            self.send_to_neighbors(self.c)
            c_j_lst = self.recv_from_neighbors()

            # Calculate the gradient
            li_c = self.c * self.in_degree - sum(c_j_lst)

            # Accelerated gradient method
            w_temp = self.w
            theta_temp = self.theta

            self.w = self.y - self.gamma * li_c
            self.theta = (1 + np.sqrt(1 + 4 * (self.theta ** 2))) / 2
            self.y = self.w + ((theta_temp - 1) / self.theta) * (self.w - w_temp)


class NodeExp2(dissys.Node):
    def __init__(self, name, edges_send, edges_recv, iterations, dim, gamma, f_i, a_i: np.ndarray, x_upper_i,
                 b_i=None):
        super().__init__(name, edges_send, edges_recv)
        # Iteration numbers
        self.T = iterations

        # Local decision variables
        self.x_i = cvx.Variable(dim)
        # Array for storing the iterations of the local decision variables
        self.x_iter = np.zeros((dim, self.T))

        # The constant in the diminishing step size for (sub)gradient method
        self.gamma = gamma

        # Local objective function and constraint coefficient matrix
        self.f_i = f_i
        self.a_i = a_i
        # The number of local constraints
        self.cons_num = a_i.shape[0]

        # Auxiliary variables y and Lagrange multipliers c
        self.y = np.zeros(self.cons_num)
        self.c = np.zeros(self.cons_num)
        # Arrays for storing the iterations of auxiliary variables y and Lagrange multipliers c
        self.y_iter = np.zeros((self.cons_num, self.T))
        self.c_iter = np.zeros((self.cons_num, self.T))

        # Array for storing the iterations of local objective function
        self.f_iter = np.zeros(self.T)

        # Model the local problem
        self.li_y = cvx.Parameter(self.cons_num)
        cons = [
            cvx.constraints.NonPos(self.a_i @ self.x_i + self.li_y - (np.zeros(self.cons_num) if b_i is None else b_i)),
            cvx.constraints.NonPos(self.x_i - x_upper_i)]
        self.prob = cvx.Problem(cvx.Minimize(self.f_i(self.x_i)), cons)

    def run(self) -> None:
        for k in range(self.T):
            # Exchange the information of y with neighbors
            self.send_to_neighbors(self.y)
            y_j_lst = self.recv_from_neighbors()

            # Update the parameters of the local problem
            self.li_y.value = self.y * self.in_degree - sum(y_j_lst)

            self.prob.solve(solver=cvx.GLPK)

            # Update the Lagrange multipliers
            self.c = self.prob.constraints[0].dual_value

            # Store the required data
            self.x_iter[:, k] = self.x_i.value
            self.f_iter[k] = self.prob.value
            self.y_iter[:, k] = self.y
            self.c_iter[:, k] = self.c

            # Exchange the information of c with neighbors
            self.send_to_neighbors(self.c)
            c_j_lst = self.recv_from_neighbors()

            # Calculate the gradient
            li_c = self.c * self.in_degree - sum(c_j_lst)

            # Subgradient method
            self.y = self.y - (self.gamma / np.sqrt(k + 1)) * li_c


def resource_perturbation(eps, delt, sens, rsrc_dim, rsrc):
    s = (sens / eps) * np.log(rsrc_dim * (np.exp(eps) - 1) / delt + 1)
    trunc_lap = tl.TruncatedLaplace(-s, s, 0, sens / eps)
    rsrc_perturbed = rsrc - s * np.ones(rsrc_dim) + trunc_lap(rsrc_dim)

    return rsrc_perturbed


def gen_nodes(exp, conn, iterations, dim, gamma, f_dic, a_dic, b_src, x_upper_dic=None):
    q_send, q_recv = dissys.gen_communication_edges(conn)
    nodes = []

    if exp == '1':
        for node in conn.keys():
            if node == '1':
                nodes.append(NodeExp1(node, q_send[node], q_recv[node],
                                      iterations, dim, gamma, f_dic[node], a_dic[node], b_i=b_src))
            else:
                nodes.append(NodeExp1(node, q_send[node], q_recv[node],
                                      iterations, dim, gamma, f_dic[node], a_dic[node]))
    elif (exp == '2') and (x_upper_dic is not None):
        for node in conn.keys():
            if node == '1':
                nodes.append(NodeExp2(node, q_send[node], q_recv[node],
                                      iterations, dim, gamma, f_dic[node], a_dic[node], x_upper_dic[node], b_i=b_src))
            else:
                nodes.append(NodeExp2(node, q_send[node], q_recv[node],
                                      iterations, dim, gamma, f_dic[node], a_dic[node], x_upper_dic[node]))

    return nodes


def save_data(iterations, nodes, f_star, a_dic, b_src, exp):
    # Error between F_iter and F_star
    f_iter = np.zeros(iterations)
    for node in nodes:
        f_iter = f_iter + node.f_iter

    err = f_iter - f_star
    df = pd.DataFrame(err)
    df.to_excel(r'..\data\experiment' + exp + r'\err.xlsx', index=False)

    # Lagrange multipliers
    for node in nodes:
        df = pd.DataFrame(node.c_iter)
        df.to_excel(r'..\data\experiment' + exp + r'\node' + node.name + r'\c_iter.xlsx', index=False)

    # Constraints' values
    cons_iter = -np.kron(np.ones(iterations), b_src.reshape(-1, 1))
    for node in nodes:
        cons_iter = cons_iter + a_dic[node.name] @ node.x_iter

    df = pd.DataFrame(cons_iter)
    df.to_excel(r'..\data\experiment' + exp + r'\cons_iter.xlsx', index=False)


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    experiment = '2'

    if experiment == '1':
        # Communication connections - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        connections = {'1': ['2'],
                       '2': ['1', '3'],
                       '3': ['2', '4', '6', '7'],
                       '4': ['3', '5'],
                       '5': ['4'],
                       '6': ['3', '8'],
                       '7': ['3', '9'],
                       '8': ['6'],
                       '9': ['7']}

        # Parameters initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 2000
        di = 4
        M = 3
        step_size = 3

        Q = {}
        g = {}
        A = {}

        # Generate matrices
        # for node in nodes:
        #     Q_i_prime = np.random.randn(di, di)
        #     Q[node] = np.eye(di) + Q_i_prime @ Q_i_prime.T
        #     df = pd.DataFrame(Q[node])
        #     df.to_excel(r'..\data\experiment1\node' + node + r'\Q.xlsx', index=False)
        #
        #     g[node] = np.random.randint(-5, 5, di)
        #     df = pd.DataFrame(g[node])
        #     df.to_excel(r'..\data\experiment1\node' + node + r'\g.xlsx', index=False)
        #
        #     A[node] = np.random.randint(-50, 50, (M, di))
        #     print(np.linalg.matrix_rank(A[node]))
        #     df = pd.DataFrame(A[node])
        #     df.to_excel(r'..\data\experiment1\node' + node + r'\A.xlsx', index=False)

        # vec = np.random.uniform(0, 1, 3)
        # pr = vec / vec.sum()
        # D = np.random.choice([1, 2, 3], 1000, p=pr)
        # df = pd.DataFrame(D)
        # df.to_excel(r'..\data\experiment1\D.xlsx', index=False)

        for node_i in connections.keys():
            Q[node_i] = pd.read_excel(r'..\data\experiment1\node' + node_i + r'\Q.xlsx').values
            g[node_i] = pd.read_excel(r'..\data\experiment1\node' + node_i + r'\g.xlsx').values.reshape(-1)
            A[node_i] = pd.read_excel(r'..\data\experiment1\node' + node_i + r'\A.xlsx').values

        D = pd.read_excel(r'..\data\experiment1\D.xlsx').values.reshape(-1)
        b = np.array(
            [D[np.where(D == 1)].size / 1000, D[np.where(D == 2)].size / 1000, D[np.where(D == 3)].size / 1000])

        f = {}
        for node_i in connections.keys():
            f[node_i] = lambda var, i=node_i: var @ Q[i] @ var / 2 + g[i] @ var

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {node_i: cvx.Variable(di) for node_i in connections.keys()}

        Cost = 0
        for node_i in connections.keys():
            Cost = Cost + f[node_i](x[node_i])

        coupled_cons = -b
        for node_i in connections.keys():
            coupled_cons = coupled_cons + A[node_i] @ x[node_i]

        constraints = [cvx.constraints.NonPos(coupled_cons)]

        prob_cen = cvx.Problem(cvx.Minimize(Cost), constraints)
        prob_cen.solve(solver=cvx.OSQP)

        F_star = prob_cen.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        b_bar = resource_perturbation(epsilon, delta, Delta, M, b)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Nodes = gen_nodes(experiment, connections, T, di, step_size, f, A, b_bar)

        for node_i in Nodes:
            node_i.start()

        for node_i in Nodes:
            node_i.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_data(T, Nodes, F_star, A, b, experiment)

    elif experiment == '2':
        # Communication connections - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        connections = {'1': ['9', '11'],
                       '2': ['3', '6', '7', '10'],
                       '3': ['2', '15'],
                       '4': ['8'],
                       '5': ['14'],
                       '6': ['2', '12'],
                       '7': ['2'],
                       '8': ['4', '15'],
                       '9': ['1'],
                       '10': ['2'],
                       '11': ['1', '13', '14'],
                       '12': ['6'],
                       '13': ['11', '15'],
                       '14': ['5', '11'],
                       '15': ['3', '8', '13']}

        # Parameters initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 3000
        di = 2
        M = 2
        step_size = 10

        c_pro = {}
        a_mat = {}
        x_lab = {}

        # Generate matrices
        # for node in nodes:
        #     c_pro[node] = np.random.uniform(1, 2, di)
        #     df = pd.DataFrame(c_pro[node])
        #     df.to_excel(r'..\data\experiment2\node' + node + r'\c_pro.xlsx', index=False)
        #
        #     a_mat[node] = np.random.uniform(1, 5, (M, di))
        #     df = pd.DataFrame(a_mat[node])
        #     df.to_excel(r'..\data\experiment2\node' + node + r'\a_mat.xlsx', index=False)
        #
        #     x_lab[node] = np.random.uniform(1, 5, di)
        #     df = pd.DataFrame(x_lab[node])
        #     df.to_excel(r'..\data\experiment2\node' + node + r'\x_lab.xlsx', index=False)
        #
        # b_mat = np.random.uniform(18, 22, M)
        # df = pd.DataFrame(b_mat)
        # df.to_excel(r'..\data\experiment2\b_mat.xlsx', index=False)

        for node_i in connections.keys():
            c_pro[node_i] = pd.read_excel(r'..\data\experiment2\node' + node_i + r'\c_pro.xlsx').values.reshape(-1)
            a_mat[node_i] = pd.read_excel(r'..\data\experiment2\node' + node_i + r'\a_mat.xlsx').values
            x_lab[node_i] = pd.read_excel(r'..\data\experiment2\node' + node_i + r'\x_lab.xlsx').values.reshape(-1)

        b_mat = pd.read_excel(r'..\data\experiment2\b_mat.xlsx').values.reshape(-1)

        f = {}
        for node_i in connections.keys():
            f[node_i] = lambda var, i=node_i: -c_pro[i] @ var

        # Centralized optimization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = {node_i: cvx.Variable(di) for node_i in connections.keys()}

        Cost = 0
        for node_i in connections.keys():
            Cost = Cost + f[node_i](x[node_i])

        mat_cons = -b_mat
        for node_i in connections.keys():
            mat_cons = mat_cons + a_mat[node_i] @ x[node_i]

        constraints = [cvx.constraints.NonPos(mat_cons)]

        for node_i in connections.keys():
            constraints.append(cvx.constraints.NonPos(x[node_i] - x_lab[node_i]))

        prob_cen = cvx.Problem(cvx.Minimize(Cost), constraints)
        prob_cen.solve(solver=cvx.GLPK)

        F_star = prob_cen.value
        print(F_star)

        # Resource perturbation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        epsilon = 0.5
        delta = 0.005
        Delta = 0.1

        b_mat_bar = resource_perturbation(epsilon, delta, Delta, M, b_mat)
        print(b_mat_bar)

        # Distributed resource allocation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Nodes = gen_nodes(experiment, connections, T, di, step_size, f, a_mat, b_mat_bar, x_lab)

        for node_i in Nodes:
            node_i.start()

        for node_i in Nodes:
            node_i.join()

        # Save the results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        save_data(T, Nodes, F_star, a_mat, b_mat, experiment)
