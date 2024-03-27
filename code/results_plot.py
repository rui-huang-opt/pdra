import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Distributed Quadratic Programming
    # Collaborative Production
    experiment = 'Distributed Quadratic Programming'

    # Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    T = {'Distributed Quadratic Programming': 2000, 'Collaborative Production': 3000}
    M = {'Distributed Quadratic Programming': 3, 'Collaborative Production': 2}
    N = {'Distributed Quadratic Programming': 9, 'Collaborative Production': 15}

    nodes = [f'{i}' for i in range(1, N[experiment] + 1)]

    # Plots - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Enable Latex rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.family'] = 'Times New Roman'

    iterations = np.arange(1, T[experiment] + 1)

    # Convergence of the algorithm
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('Iteration number k', fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)

    err = pd.read_excel(r'..\data\\' + experiment + r'\err.xlsx').values.reshape(-1)

    ax1_label = {'Distributed Quadratic Programming': r'$F(\boldsymbol{x}_k)-F(\boldsymbol{x}^*)$',
                 'Collaborative Production': r'$P(\boldsymbol{x}^*)-P(\boldsymbol{x}_k)$'}

    ax1.semilogy(iterations, err, label=ax1_label[experiment])

    ax1.legend(loc='upper right', fontsize=15)

    # fig1.savefig(r"..\manuscript\src\figures\fig3_a.png", dpi=300, bbox_inches='tight')

    # Lagrange multipliers
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_xlabel('Iteration number k', fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)

    c_iter = {node: pd.read_excel(r'..\data\\' + experiment + r'\node' + node + r'\c_iter.xlsx').values
              for node in nodes}

    if experiment == 'Distributed Quadratic Programming':
        color = {'1': 'tab:blue', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:red', '5': 'tab:purple',
                 '6': 'tab:brown', '7': 'tab:pink', '8': 'tab:gray', '9': 'tab:olive'}

        for node in nodes:
            for m in range(M[experiment]):
                if m == 1:
                    ax2.step(iterations, c_iter[node][m, :], color=color[node], label='agent ' + node)
                else:
                    ax2.step(iterations, c_iter[node][m, :], color=color[node])

        ax2.legend(loc='upper right', fontsize=15)

    elif experiment == 'Collaborative Production':
        ax2.set_ylim(0, 6)

        ax2ins = ax2.inset_axes((0.2, 0.3, 0.6, 0.6))
        ax2ins.tick_params(axis='both', labelsize=15)

        for node in nodes:
            for m in range(M[experiment]):
                ax2.step(iterations, c_iter[node][m, :])
                ax2ins.step(iterations[2950:2999], c_iter[node][m, 2950:2999])

    # fig2.savefig(r"..\manuscript\src\figures\fig3_b.png", dpi=300, bbox_inches='tight')

    # The iterations of the coupling constraints
    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_xlabel('Iteration number k', fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)

    cons_iter = pd.read_excel(r'..\data\\' + experiment + r'\cons_iter.xlsx').values

    ax3_label = {'Distributed Quadratic Programming': 'constraint ', 'Collaborative Production': 'material '}

    for m in range(M[experiment]):
        ax3.step(iterations, cons_iter[m, :], label=ax3_label[experiment] + f'{m + 1}')

    ax3.legend(loc='lower right', fontsize=15)

    # fig3.savefig(r"..\manuscript\src\figures\fig3_c.png", dpi=300, bbox_inches='tight')

    plt.show()
