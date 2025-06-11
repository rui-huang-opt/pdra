import numpy as np
import matplotlib.pyplot as plt
import toml
from pdra import TruncatedLaplace

if __name__ == "__main__":
    config = toml.load("configs/truncated_laplace.toml")

    tl = TruncatedLaplace(config["low"], config["high"], config["loc"], config["scale"])

    samples = tl.sample(config["n_samples"])

    figure_path = "figures/truncated_laplace/"

    fig, ax = plt.subplots()

    ax.hist(samples, bins=config["n_bins"], density=True)

    x = np.linspace(config["low"], config["high"], config["n_points"])

    ax.plot(x, tl.pdf(x), "r", lw=2)

    ax.set_title("Truncated Laplace Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("pdf(x)")

    fig.savefig(figure_path + "truncated_laplace.png", dpi=300, bbox_inches="tight")
