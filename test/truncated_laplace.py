import numpy as np
import matplotlib.pyplot as plt
from pdra import TruncatedLaplace

if __name__ == "__main__":
    low = -2
    high = 1
    loc = 0
    scale = 1

    n_samples = 50000

    tl = TruncatedLaplace(low, high, loc, scale)

    samples = tl(n_samples)

    figure_path = "figures/truncated_laplace/"

    fig, ax = plt.subplots()

    ax.hist(samples, bins=50, density=True)

    x = np.linspace(low, high, 1000)

    ax.plot(x, tl.pdf(x), "r", lw=2)

    ax.set_title("Truncated Laplace Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("pdf(x)")

    fig.savefig(figure_path + "truncated_laplace.png", dpi=300, bbox_inches="tight")
