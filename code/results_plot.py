import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Distributed Quadratic Programming
    # Collaborative Production
    experiment = 'Collaborative Production'

    # Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    T = {'Distributed Quadratic Programming': 2000, 'Collaborative Production': 3000}
    M = {'Distributed Quadratic Programming': 3, 'Collaborative Production': 2}
    N = {'Distributed Quadratic Programming': 9, 'Collaborative Production': 15}

    nodes = [f'{i}' for i in range(1, N[experiment] + 1)]

    # Plots - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    iterations = np.arange(1, T[experiment] + 1)

    # Convergence of the algorithm
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('Iteration number k', fontsize=15)

    err = pd.read_excel(r'..\data\\' + experiment + r'\err.xlsx').values.reshape(-1)

    if experiment == 'Distributed Quadratic Programming':
        ax1.semilogy(iterations, err, label='$F(x_k)-F(x^*)$')

        ax1ins = ax1.inset_axes((0.4, 0.4, 0.5, 0.3))
        ax1ins.set_ylim(0, 0.005)
        ax1ins.step(iterations[1800:1999], err[1800:1999])
    elif experiment == 'Collaborative Production':
        ax1.semilogy(iterations, err, label='$P(x^*)-P(x_k)$')

        ax1ins = ax1.inset_axes((0.3, 0.35, 0.5, 0.5))
        ax1ins.set_ylim(5, 10)
        ax1ins.step(iterations[2500:2999], err[2500:2999])

    ax1.legend(loc='upper right')

    # fig1.savefig(r"..\manuscript\src\figures\fig4_a.png", dpi=300, bbox_inches='tight')

    # Lagrange multipliers
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_xlabel('Iteration number k', fontsize=15)

    c_iter = {node: pd.read_excel(r'..\data\\' + experiment + r'\node' + node + r'\c_iter.xlsx').values
              for node in nodes}

    if experiment == 'Distributed Quadratic Programming':
        color = {'1': 'tab:blue', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:red', '5': 'tab:purple',
                 '6': 'tab:brown', '7': 'tab:pink', '8': 'tab:gray', '9': 'tab:olive'}

        for node in nodes:
            for m in range(M[experiment]):
                if m == 1:
                    ax2.step(iterations, c_iter[node][m, :], color=color[node], label='node ' + node)
                else:
                    ax2.step(iterations, c_iter[node][m, :], color=color[node])

        ax2.legend(loc='upper right')

    elif experiment == 'Collaborative Production':
        ax2.set_ylim(0, 6)

        ax2ins = ax2.inset_axes((0.2, 0.3, 0.6, 0.6))

        for node in nodes:
            for m in range(M[experiment]):
                ax2.step(iterations, c_iter[node][m, :])
                ax2ins.step(iterations[2950:2999], c_iter[node][m, 2950:2999])

    # fig2.savefig(r"..\manuscript\src\figures\fig4_b.png", dpi=300, bbox_inches='tight')

    # The iterations of the coupling constraints
    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_xlabel('Iteration number k', fontsize=15)

    cons_iter = pd.read_excel(r'..\data\\' + experiment + r'\cons_iter.xlsx').values

    ax3ins = ax3.inset_axes((0.3, 0.3, 0.5, 0.5))

    if experiment == 'Distributed Quadratic Programming':
        for m in range(M[experiment]):
            ax3.step(iterations, cons_iter[m, :], label=f'constraint {m + 1}')
            ax3ins.step(iterations[1800:1999], cons_iter[m, 1800:1999])

    elif experiment == 'Collaborative Production':
        for m in range(M[experiment]):
            ax3.step(iterations, cons_iter[m, :], label=f'material {m + 1}')
            ax3ins.step(iterations[2500:2999], cons_iter[m, 2500:2999])

    ax3.legend(loc='lower right')

    # fig3.savefig(r"..\manuscript\src\figures\fig4_c.png", dpi=300, bbox_inches='tight')

    plt.show()
