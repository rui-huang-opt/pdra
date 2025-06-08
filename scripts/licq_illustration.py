import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def calculate_vertices(
    lhs: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64]:
    if lhs.ndim != 2 or lhs.shape[1] != 2:
        raise ValueError("lhs must be a 2D array with shape (n_constraints, 2).")
    if rhs.ndim != 1 or lhs.shape[0] != rhs.shape[0]:
        raise ValueError(
            "rhs must be a 1D array with the same length as the number of constraints in lhs."
        )

    angles = np.arctan2(lhs[:, 1], lhs[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_lhs = lhs[sorted_indices]
    sorted_rhs = rhs[sorted_indices]

    num_constraints = lhs.shape[0]
    vertices: list[NDArray[np.float64]] = []
    for i in range(num_constraints):
        j = (i + 1) % num_constraints
        temp_lhs = np.vstack((sorted_lhs[i], sorted_lhs[j]))
        temp_rhs = np.hstack((sorted_rhs[i], sorted_rhs[j]))
        try:
            vertex = np.linalg.solve(temp_lhs, temp_rhs)
            if np.all(lhs @ vertex <= rhs):
                vertices.append(vertex)
        except np.linalg.LinAlgError:
            continue

    return np.vstack(vertices)


def plot_constraints(
    lhs: NDArray[np.float64],
    rhs: NDArray[np.float64],
    ax: plt.Axes,
    fig_name: str,
    x_lim: tuple[float, float] = (-3, 3),
    y_lim: tuple[float, float] = (-2, 2),
    step: float = 0.1,
) -> None:
    x = np.arange(x_lim[0], x_lim[1] + step, step)

    vertices = calculate_vertices(lhs, rhs)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    for i in range(A.shape[0]):
        a_i = A[i]
        b_i = b[i]
        terms = []
        # x1 term
        if a_i[0] != 0:
            if a_i[0] == 1:
                terms.append("x_1")
            elif a_i[0] == -1:
                terms.append("-x_1")
            else:
                terms.append(f"{a_i[0]:g}x_1")
        # x2 term
        if a_i[1] != 0:
            sign = "+" if a_i[1] > 0 and terms else ""
            if a_i[1] == 1:
                terms.append(f"{sign}x_2")
            elif a_i[1] == -1:
                terms.append(f"{sign}-x_2")
            else:
                terms.append(f"{sign}{a_i[1]:g}x_2")
        lhs_str = " ".join(terms) if terms else "0"
        constraint_label = f"${lhs_str} \\leq {b_i:g}$"
        ax.plot(x, (b_i - a_i[0] * x) / a_i[1], label=constraint_label)

    ax.fill(
        vertices[:, 0],
        vertices[:, 1],
        alpha=0.3,
        color="gray",
        label="Feasible Region",
    )
    ax.plot(vertices[:, 0], vertices[:, 1], "ro", label="Vertices")

    ax.set_title("LICQ Illustration")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_aspect("equal")

    ax.legend()

    fig.savefig(f"{fig_name}", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fig_name}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figure_dir = os.path.join(root_dir, "figures", "licq")
    os.makedirs(figure_dir, exist_ok=True)

    A = np.array([[-1.0, -2.0], [1.0, 1.0], [-1.0, 2.0], [0.0, -1.0]])
    b = np.array([2.0, 1.0, 2.0, 1.0])

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_constraints(A, b, ax, os.path.join(figure_dir, "licq_illustration_1.png"))

    b[-1] = 0.0
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_constraints(A, b, ax, os.path.join(figure_dir, "licq_illustration_2.png"))

    b[-1] = -1.0
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_constraints(A, b, ax, os.path.join(figure_dir, "licq_illustration_3.png"))
