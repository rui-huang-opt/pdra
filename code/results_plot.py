import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    experiment = '2'

    if experiment == '1':
        # Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        T = 2000
        di = 4
        M = 3

        N = 9
        nodes = [f'{i}' for i in range(1, N + 1)]

        # The results of the experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        iterations = np.arange(1, T + 1)

        # Convergence of the algorithm
        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_xlabel('Iteration number k')

        err = pd.read_excel(r'..\data\experiment1\err.xlsx').values.reshape(-1)

        ax1.semilogy(iterations, err, label='$F(x_k)-F(x^*)$')

        ax1ins = ax1.inset_axes((0.4, 0.4, 0.5, 0.3))
        ax1ins.set_ylim(0, 0.005)
        ax1ins.step(iterations[1800:1999], err[1800:1999])

        ax1.legend(loc='upper right')

        # plt.savefig(r"..\manuscript\src\figures\fig3_a.png", dpi=300, bbox_inches='tight')

        # Lagrange multipliers
        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_xlabel('Iteration number k')
        color = {'1': 'tab:blue', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:red', '5': 'tab:purple',
                 '6': 'tab:brown', '7': 'tab:pink', '8': 'tab:gray', '9': 'tab:olive'}

        for node in nodes:
            c_iter = pd.read_excel(r'..\data\experiment1\node' + node + r'\c_iter.xlsx').values

            for m in range(M):
                if m == 1:
                    ax2.step(iterations, c_iter[m, :], color=color[node], label='node ' + node)
                else:
                    ax2.step(iterations, c_iter[m, :], color=color[node])

        ax2.legend(loc='upper right')

        # plt.savefig(r"..\manuscript\src\figures\fig3_b.png", dpi=300, bbox_inches='tight')

        # The iterations of the coupling constraints
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_xlabel('Iteration number k')

        cons_iter = pd.read_excel(r'..\data\experiment1\cons_iter.xlsx').values

        for m in range(M):
            ax3.step(iterations, cons_iter[m, :], label=f'constraint {m + 1}')

        ax3ins = ax3.inset_axes((0.3, 0.3, 0.5, 0.5))
        for m in range(M):
            ax3ins.step(iterations[1800:1999], cons_iter[m, 1800:1999])

        ax3.legend(loc='lower right')

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

        # The results of the experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        iterations = np.arange(1, T + 1)

        # Convergence of the algorithm
        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_xlabel('Iteration number k')

        err = pd.read_excel(r'..\data\experiment2\err.xlsx').values.reshape(-1)

        ax1.semilogy(iterations, err, label='$P(x^*)-P(x_k)$')

        ax1ins = ax1.inset_axes((0.3, 0.35, 0.5, 0.5))
        ax1ins.set_ylim(5, 10)
        ax1ins.step(iterations[2500:2999], err[2500:2999])

        ax1.legend(loc='upper right')

        # plt.savefig(r"..\manuscript\src\figures\fig4_a.png", dpi=300, bbox_inches='tight')

        # Lagrange multipliers
        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_ylim(0, 6)
        ax2.set_xlabel('Iteration number k')

        ax2ins = ax2.inset_axes((0.2, 0.3, 0.6, 0.6))

        for node in nodes:
            c_iter = pd.read_excel(r'..\data\experiment2\node' + node + r'\c_iter.xlsx').values

            for m in range(M):
                ax2.step(iterations, c_iter[m, :])
                ax2ins.step(iterations[2950:2999], c_iter[m, 2950:2999])

        # plt.savefig(r"..\manuscript\src\figures\fig4_b.png", dpi=300, bbox_inches='tight')

        # The iterations of the coupling constraints
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_xlabel('Iteration number k')

        cons_iter = pd.read_excel(r'..\data\experiment2\cons_iter.xlsx').values

        for m in range(M):
            ax3.step(iterations, cons_iter[m, :], label=f'material {m + 1}')

        ax3ins = ax3.inset_axes((0.3, 0.3, 0.5, 0.5))
        for m in range(M):
            ax3ins.step(iterations[2500:2999], cons_iter[m, 2500:2999])

        ax3.legend(loc='lower right')

        # plt.savefig(r"..\manuscript\src\figures\fig4_c.png", dpi=300, bbox_inches='tight')

        plt.show()
