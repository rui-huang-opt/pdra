import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Experiment - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Distributed Quadratic Programming
    # Collaborative Production
    experiment = 'Collaborative Production'

    # Load data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    err_series = np.load(rf'..\data\{experiment}\results\err_series.npy')
    c_series = np.load(rf'..\data\{experiment}\results\c_series.npz')
    cons_series = np.load(rf'..\data\{experiment}\results\cons_series.npy')
    iterations = np.arange(1, err_series.size + 1)

    # Plots - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Enable Latex rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.family'] = 'Times New Roman'

    # Convergence of the algorithm
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('Iteration number $k$', fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)

    ax1_label = {'Distributed Quadratic Programming': r'$F(\boldsymbol{x}_k)-F(\boldsymbol{x}^*)$',
                 'Collaborative Production': r'$P(\boldsymbol{x}^*)-P(\boldsymbol{x}_k)$'}

    ax1.step(iterations, err_series, label=ax1_label[experiment])

    ax1.legend(loc='upper right', fontsize=15)

    # fig1.savefig(r"..\manuscript\src\figures\fig3_a.png", dpi=300, bbox_inches='tight')

    # Lagrange multipliers
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_xlabel('Iteration number $k$', fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)

    if experiment == 'Distributed Quadratic Programming':
        color = {'1': 'tab:blue', '2': 'tab:orange', '3': 'tab:green', '4': 'tab:red', '5': 'tab:purple',
                 '6': 'tab:brown', '7': 'tab:pink', '8': 'tab:gray', '9': 'tab:olive'}

        for i in sorted(c_series):
            for m in range(c_series[i].shape[0]):
                if m == 1:
                    ax2.step(iterations, c_series[i][m, :], color=color[i], label=f'agent {i}')
                else:
                    ax2.step(iterations, c_series[i][m, :], color=color[i])

        ax2.legend(loc='upper right', fontsize=15)

    elif experiment == 'Collaborative Production':
        ax2.set_ylim(0, 6)

        ax2ins = ax2.inset_axes((0.2, 0.3, 0.6, 0.6))
        ax2ins.tick_params(axis='both', labelsize=15)

        for i in sorted(c_series):
            for m in range(c_series[i].shape[0]):
                ax2.step(iterations, c_series[i][m, :])
                ax2ins.step(iterations[2950:2999], c_series[i][m, 2950:2999])

    # fig2.savefig(r"..\manuscript\src\figures\fig3_b.png", dpi=300, bbox_inches='tight')

    # The iterations of the coupling constraints
    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_xlabel('Iteration number $k$', fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)

    ax3_label = {'Distributed Quadratic Programming': 'constraint', 'Collaborative Production': 'material'}

    for m in range(cons_series.shape[0]):
        ax3.step(iterations, cons_series[m, :], label=f'{ax3_label[experiment]} {m + 1}')

    ax3.legend(loc='lower right', fontsize=15)

    # fig3.savefig(r"..\manuscript\src\figures\fig3_c.png", dpi=300, bbox_inches='tight')

    plt.show()
