import os
import toml
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from functools import partial
from gossip import create_sync_network
from pdra import Node, TruncatedLaplace

if __name__ == "__main__":
    # Load the configuration
    configs = toml.load("../configs.toml")["dqp"]

    node_names: list[str] = configs["node_names"]
    edge_pairs: list[tuple[str, str]] = configs["edge_pairs"]

    algorithm = configs["algorithm"]
    node_params = configs["node_params"]

    # Load the model of the problem
    ## Generate the model as described in the paper
    # di = 4
    # M = 3
    #
    # Q = {}
    # g = {}
    # A = {}
    #
    # for i in NODES:
    #     Q_i_prime = np.random.randn(di, di)
    #     Q[i] = np.eye(di) + Q_i_prime @ Q_i_prime.T
    #     g[i] = np.random.randint(-5, 5, di)
    #     A[i] = np.random.randint(-50, 50, (M, di))
    #
    # np.savez(configs["data_dir"] + "Q.npz", **Q)
    # np.savez(configs["data_dir"] + "g.npz", **g)
    # np.savez(configs["data_dir"] + "A.npz", **A)
    #
    # vec = np.random.uniform(0, 1, 3)
    # pr = vec / vec.sum()
    # D = np.random.choice([1, 2, 3], 1000, p=pr)
    # np.save(configs["data_path"] + "D.npy", D)

    Q: Dict[str, NDArray[np.float64]] = np.load(configs["data_dir"] + "Q.npz")
    g: Dict[str, NDArray[np.float64]] = np.load(configs["data_dir"] + "g.npz")
    A: Dict[str, NDArray[np.float64]] = np.load(configs["data_dir"] + "A.npz")

    D: NDArray[np.float64] = np.load(configs["data_dir"] + "D.npy")

    proportion_of_1 = D[np.where(D == 1)].size / 1000
    proportion_of_2 = D[np.where(D == 2)].size / 1000
    proportion_of_3 = D[np.where(D == 3)].size / 1000

    b = np.array([proportion_of_1, proportion_of_2, proportion_of_3])

    if configs["run_type"] == "cen":
        # Centralized optimization
        x: dict[str, cp.Variable] = {i: cp.Variable(A[i].shape[1]) for i in node_names}

        cost = cp.sum([x[i] @ Q[i] @ x[i] / 2 + g[i] @ x[i] for i in node_names])
        constraints: list[cp.Constraint] = [
            sum([A[i] @ x[i] for i in node_names], start=cp.Constant(0)) <= b
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver="OSQP")

        opt_val = problem.value

        print(f"Centralized optimal value: {opt_val}")

        with open(f"../configs.toml", "w") as file:
            configs["opt_val"] = opt_val
            toml.dump(configs, file)

    elif configs["run_type"] == "sen":
        # Sensitivity analysis for the differential privacy
        opt_val = configs["opt_val"]

        x: dict[str, cp.Variable] = {i: cp.Variable(A[i].shape[1]) for i in node_names}
        b_bar = cp.Parameter(b.size)

        cost = cp.sum([x[i] @ Q[i] @ x[i] / 2 + g[i] @ x[i] for i in node_names])
        constraints: list[cp.Constraint] = [
            sum([A[i] @ x[i] for i in node_names], start=cp.Constant(0)) <= b_bar
        ]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        epsilons = np.arange(0.05, 0.525, 0.025)
        deltas = np.arange(1e-5, 0.0042 + 1e-5, 0.0002)
        Delta = 0.002

        n_eps = epsilons.size
        n_del = deltas.size
        n_samples = n_eps * n_del
        run_times = 50

        Eps, Del = np.meshgrid(epsilons, deltas, indexing="ij")
        param_pairs = np.dstack((Eps, Del)).reshape(-1, 2)

        opt_perturb = np.zeros((n_samples, run_times))

        for i in range(n_samples):
            epsilon, delta = param_pairs[i]
            s = (Delta / epsilon) * np.log(b.size * (np.exp(epsilon) - 1) / delta + 1)
            tl = TruncatedLaplace(-s, s, 0, Delta / epsilon)

            perturbation = -s * np.ones(b.size) + tl.sample(b.size)
            b_bar.value = b + perturbation

            for j in range(run_times):
                problem.solve(solver="OSQP")
                opt_perturb[i, j] = problem.value

        rel_opt_gap: NDArray[np.float64] = np.abs(
            (np.mean(opt_perturb, axis=1) - opt_val) / opt_val
        )
        rel_opt_gap = rel_opt_gap.reshape(n_eps, n_del)

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["font.size"] = 15

        fig, ax = plt.subplots(figsize=(10, 6))

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=epsilons.min(), vmax=epsilons.max())

        for i in range(n_eps):
            color = cmap(norm(epsilons[i]))
            ax.plot(
                deltas,
                rel_opt_gap[i, :],
                label=f"$\\epsilon = {epsilons[i]:.2f}$",
                color=color,
            )

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("$\\epsilon$")
        ax.set_xlim(0.0, 0.004)
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.set_xlabel("$\\delta$")
        ax.set_ylabel("Relative optimality gap")

        fig.savefig(
            os.path.join(configs["fig_dir"], "sensitivity_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            os.path.join(configs["fig_dir"], "sensitivity_analysis.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    elif configs["run_type"] == "dis":
        # Resource perturbation
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        np.random.seed(0)

        if algorithm == "core":
            s = (Delta / epsilon) * np.log(b.size * (np.exp(epsilon) - 1) / delta + 1)
            tl = TruncatedLaplace(-s, s, 0, Delta / epsilon)
            perturbation = -s * np.ones(b.size) + tl.sample(b.size)
        elif algorithm == "rsdd":
            perturbation = np.random.laplace(0, Delta / epsilon, b.size)
        else:
            raise ValueError("Invalid algorithm")

        b_bar = b + perturbation

        # Distributed resource allocation
        gossip_network = create_sync_network(node_names, edge_pairs)

        def f(x: cp.Variable, index: str) -> cp.Expression:
            return x @ Q[index] @ x / 2 + g[index] @ x

        nodes = [
            Node.create(
                algorithm,
                gossip_network[i],
                partial(f, index=i),
                A[i],
                **node_params[algorithm],
            )
            for i in node_names
        ]

        nodes[0].set_resource(b_bar)

        for node in nodes:
            node.start()

        for node in nodes:
            node.join()

    elif configs["run_type"] == "plot":
        fig_dir = configs["fig_dir"]
        os.makedirs(fig_dir, exist_ok=True)

        # Plot the graph and the result
        graph = nx.Graph()
        graph.add_nodes_from(node_names)
        graph.add_edges_from(edge_pairs)

        fig1, ax1 = plt.subplots(1, 1)

        ax1.set_aspect(1)
        ax1.set_ylim(-1.5, 1.5)

        nx.draw(graph, ax=ax1, **configs["nx_options"])

        fig1.savefig(
            os.path.join(configs["fig_dir"], "fig_1.png"), dpi=300, bbox_inches="tight"
        )
        fig1.savefig(
            os.path.join(configs["fig_dir"], "fig_1.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["font.size"] = 15

        results = {
            "core": {
                i: np.load(node_params["core"]["results_prefix"] + f"/node_{i}.npz")
                for i in node_names
            },
            "rsdd": {
                i: np.load(node_params["rsdd"]["results_prefix"] + f"/node_{i}.npz")
                for i in node_names
            },
        }

        iterations = np.arange(1, node_params["core"]["max_iter"] + 1)

        fig2, ax2 = plt.subplots()

        err_series = {
            "core": sum([results["core"][i]["f_i_series"] for i in node_names])
            - configs["opt_val"],
            "rsdd": sum([results["rsdd"][i]["f_i_series"] for i in node_names])
            - configs["opt_val"],
        }

        ax2.step(
            iterations,
            err_series["core"],
            label="The proposed algorithm",
            color="tab:blue",
            linestyle="-",
        )

        ax2.step(
            iterations,
            err_series["rsdd"],
            label="RSDD",
            color="tab:orange",
            linestyle="--",
        )

        ax2.set_xlabel("Iteration number $k$")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid()

        fig2.savefig(
            os.path.join(configs["fig_dir"], "fig_3_a.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(configs["fig_dir"], "fig_3_a.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

        colors = {
            "1": "tab:blue",
            "2": "tab:orange",
            "3": "tab:green",
            "4": "tab:red",
            "5": "tab:purple",
            "6": "tab:brown",
            "7": "tab:pink",
            "8": "tab:gray",
            "9": "tab:olive",
        }

        fig3, ax3 = plt.subplots()

        c_series: Dict[str, NDArray[np.float64]] = {
            i: results["core"][i]["c_i_series"] for i in node_names
        }

        for i in node_names:
            for j in range(c_series[i].shape[0]):
                (line,) = ax3.step(iterations, c_series[i][j], color=colors[i])

            line.set_label(f"node {i}")

        ax3.set_xlabel("Iteration number $k$")
        ax3.legend()
        ax3.grid()

        fig3.savefig(
            os.path.join(configs["fig_dir"], "fig_3_b.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(configs["fig_dir"], "fig_3_b.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

        fig4, ax4 = plt.subplots()

        constraint_values = {
            "core": sum([A[i] @ results["core"][i]["x_i_series"] for i in node_names])
            - b[:, np.newaxis],
            "rsdd": sum([A[i] @ results["rsdd"][i]["x_i_series"] for i in node_names])
            - b[:, np.newaxis],
        }

        for i in range(constraint_values["core"].shape[0]):
            ax4.step(
                iterations,
                constraint_values["core"][i],
                linestyle="-",
                color=colors[str(i + 1)],
                label=f"The proposed algorithm, constraint {i + 1}",
            )
            ax4.step(
                iterations,
                constraint_values["rsdd"][i],
                linestyle="--",
                color=colors[str(i + 1)],
                label=f"rsdd, constraint {i + 1}",
            )

        ax4.set_xlabel("Iteration number $k$")
        ax4.legend()
        ax4.grid()

        fig4.savefig(
            os.path.join(configs["fig_dir"], "fig_3_c.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(configs["fig_dir"], "fig_3_c.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

        computation_times = {
            "core": np.vstack(
                [results["core"][i]["computation_time"] for i in node_names]
            ),
            "rsdd": np.vstack(
                [results["rsdd"][i]["computation_time"] for i in node_names]
            ),
        }

        avg_times = {
            "core": np.mean(computation_times["core"], axis=0),
            "rsdd": np.mean(computation_times["rsdd"], axis=0),
        }

        max_times = {
            "core": np.max(computation_times["core"], axis=0),
            "rsdd": np.max(computation_times["rsdd"], axis=0),
        }

        fig5, ax5 = plt.subplots()

        ax5.step(
            iterations,
            avg_times["core"],
            label="The proposed algorithm",
            color="tab:blue",
            linestyle="-",
        )

        ax5.step(
            iterations,
            avg_times["rsdd"],
            label="RSDD",
            color="tab:orange",
            linestyle="--",
        )

        ax5.set_xlabel("Iteration number $k$")
        ax5.set_ylabel("Time (s)")
        ax5.legend()
        ax5.grid()
        fig5.savefig(
            os.path.join(configs["fig_dir"], "dqp_avg_time.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig5.savefig(
            os.path.join(configs["fig_dir"], "dqp_avg_time.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

        fig6, ax6 = plt.subplots()

        ax6.step(
            iterations,
            max_times["core"],
            label="The proposed algorithm",
            color="tab:blue",
            linestyle="-",
        )

        ax6.step(
            iterations,
            max_times["rsdd"],
            label="RSDD",
            color="tab:orange",
            linestyle="--",
        )

        ax6.set_xlabel("Iteration number $k$")
        ax6.set_ylabel("Time (s)")
        ax6.legend()
        ax6.grid()
        fig6.savefig(
            os.path.join(configs["fig_dir"], "dqp_max_time.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig6.savefig(
            os.path.join(configs["fig_dir"], "dqp_max_time.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
