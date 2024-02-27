import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    experiment = '2'

    if experiment == '1':
        # Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 2000
        di = 4
        M = 3

        # Communication graph - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        N = 9  # 节点个数

        nodes = [f'{i}' for i in range(1, N + 1)]
        edges = [('1', '2'), ('2', '3'), ('3', '4'), ('3', '6'), ('3', '7'), ('4', '5'), ('6', '8'), ('7', '9')]
        nodes_pos = {'1': (-2, 0.6), '2': (-1, 0.3), '3': (0, 0), '4': (-1, -0.3), '5': (-2, -0.6), '6': (1, 0.3),
                     '7': (1, -0.3), '8': (2, 0.6), '9': (2, -0.6)}

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        L = nx.laplacian_matrix(G).toarray()  # Laplacian matrix

        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_aspect(1)
        nx.draw(G, pos=nodes_pos, with_labels=True, ax=ax1)
        # plt.savefig(r'..\manuscript\src\figures\fig1.png', dpi=300, bbox_inches='tight')

        # The results of the experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        iter_num = np.arange(1, T + 1)

        # Convergence of the algorithm
        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_xlabel('Iteration number k')

        err = pd.read_excel(r'..\data\experiment1\err.xlsx').values.reshape(-1)

        ax2.semilogy(iter_num, err, label='$F(x_k)-F(x^*)$')

        ax2ins = ax2.inset_axes((0.4, 0.4, 0.5, 0.3))
        ax2ins.set_ylim(0, 0.005)
        ax2ins.step(iter_num[1800:1999], err[1800:1999])

        ax2.legend(loc='upper right')

        # plt.savefig(r"..\manuscript\src\figures\fig3_a.png", dpi=300, bbox_inches='tight')

        # Lagrange multipliers
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_xlabel('Iteration number k')
        color = {'1': 'tab:blue', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:red', '5': 'tab:purple',
                 '6': 'tab:brown', '7': 'tab:pink', '8': 'tab:gray', '9': 'tab:olive'}

        for node in nodes:
            c_iter = pd.read_excel(r'..\data\experiment1\node' + node + r'\c_iter.xlsx').values

            for m in range(M):
                if m == 1:
                    ax3.step(iter_num, c_iter[m, :], color=color[node], label='node ' + node)
                else:
                    ax3.step(iter_num, c_iter[m, :], color=color[node])

        ax3.legend(loc='upper right')

        # plt.savefig(r"..\manuscript\src\figures\fig3_b.png", dpi=300, bbox_inches='tight')

        # The iterations of the coupling constraints
        fig4, ax4 = plt.subplots(1, 1)
        ax4.set_xlabel('Iteration number k')

        cons_iter = pd.read_excel(r'..\data\experiment1\cons_iter.xlsx').values

        for m in range(M):
            ax4.step(iter_num, cons_iter[m, :], label=f'constraint {m + 1}')

        ax4ins = ax4.inset_axes((0.3, 0.3, 0.5, 0.5))
        for m in range(M):
            ax4ins.step(iter_num[1800:1999], cons_iter[m, 1800:1999])

        ax4.legend(loc='lower right')

        # plt.savefig(r"..\manuscript\src\figures\fig3_c.png", dpi=300, bbox_inches='tight')

        plt.show()

    elif experiment == '2':
        # Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 3000
        di = 2
        M = 2

        # Communication Graph - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        N = 15
        nodes = [f'{i}' for i in range(1, N + 1)]
        edges = [('1', '9'), ('1', '11'), ('2', '3'), ('2', '6'), ('2', '7'), ('2', '10'), ('6', '12'),
                 ('3', '15'), ('4', '8'), ('5', '14'), ('8', '15'), ('11', '13'), ('11', '14'), ('13', '15')]
        nodes_pos = {'1': [0.13436424411240122, 0.8474337369372327], '2': [0.763774618976614, 0.2550690257394217],
                     '3': [0.59543508709194095, 0.4494910647887381], '4': [0.651592972722763, 0.7887233511355132],
                     '5': [0.0938595867742349, 0.02834747652200631], '6': [0.8357651039198697, 0.43276706790505337],
                     '7': [0.762280082457942, 0.0021060533511106927], '8': [0.4453871940548014, 0.7215400323407826],
                     '9': [0.22876222127045265, 0.9452706955539223], '10': [0.9014274576114836, 0.030589983033553536],
                     '11': [0.0254458609934608, 0.5414124727934966], '12': [0.9391491627785106, 0.38120423768821243],
                     '13': [0.21659939713061338, 0.4221165755827173], '14': [0.029040787574867943, 0.22169166627303505],
                     '15': [0.43788759365057206, 0.49581224138185065]}

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        fig1, ax1 = plt.subplots(1, 1)
        nx.draw(G, pos=nodes_pos, with_labels=True, ax=ax1)
        # plt.savefig(r'..\manuscript\src\figures\fig2.png', dpi=300, bbox_inches='tight')

        # The results of the experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        iter_num = np.arange(1, T + 1)

        # Convergence of the algorithm
        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_xlabel('Iteration number k')

        err = pd.read_excel(r'..\data\experiment2\err.xlsx').values.reshape(-1)

        ax2.semilogy(iter_num, err, label='$P(x^*)-P(x_k)$')

        ax2ins = ax2.inset_axes((0.3, 0.35, 0.5, 0.5))
        ax2ins.set_ylim(5, 10)
        ax2ins.step(iter_num[2500:2999], err[2500:2999])

        ax2.legend(loc='upper right')

        # plt.savefig(r"..\manuscript\src\figures\fig4_a.png", dpi=300, bbox_inches='tight')

        # Lagrange multipliers
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_ylim(0, 6)
        ax3.set_xlabel('Iteration number k')

        ax3ins = ax3.inset_axes((0.2, 0.3, 0.6, 0.6))

        for node in nodes:
            c_iter = pd.read_excel(r'..\data\experiment2\node' + node + r'\c_iter.xlsx').values

            for m in range(M):
                ax3.step(iter_num, c_iter[m, :])
                ax3ins.step(iter_num[2950:2999], c_iter[m, 2950:2999])

        # plt.savefig(r"..\manuscript\src\figures\fig4_b.png", dpi=300, bbox_inches='tight')

        # The iterations of the coupling constraints
        fig4, ax4 = plt.subplots(1, 1)
        ax4.set_xlabel('Iteration number k')

        cons_iter = pd.read_excel(r'..\data\experiment2\cons_iter.xlsx').values

        for m in range(M):
            ax4.step(iter_num, cons_iter[m, :], label=f'material {m + 1}')

        ax4ins = ax4.inset_axes((0.3, 0.3, 0.5, 0.5))
        for m in range(M):
            ax4ins.step(iter_num[2500:2999], cons_iter[m, 2500:2999])

        ax4.legend(loc='lower right')

        # plt.savefig(r"..\manuscript\src\figures\fig4_c.png", dpi=300, bbox_inches='tight')

        plt.show()
