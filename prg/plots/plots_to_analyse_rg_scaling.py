import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.axes._axes import Axes
import scipy.odr as odr
from scipy.stats import binom
from scipy.optimize import curve_fit
from scipy.stats import moment as sp_moment


# Power law function
def power_law(x, b, a):
    return a * np.power(x, b)

# Define a wrapper function that only takes 'b' as a parameter
def power_law_fixed_a(x, b, y):
    a = y[0]
    return power_law(x, b, a)

def fit_power_law_fixed_a(x, y):
    params, pcov = curve_fit(lambda x, b: power_law_fixed_a(x, b, y), x, y)
    return params, pcov

def fit_power_law(x, y):
    p0 = [1, y[0]]
    params, pcov = curve_fit(power_law, x, y, p0=p0)
    return params, pcov

def projection(x, k, u):
    """
    Project the data x onto its k largest principal components. 
    
    Parameters:
        x - data to project
        k - number of eigenmodes to keep
        u - Eigenvectors x ordered from largest to smallest
    """
    # Check that k is smaller than the number of eigenvectors
    if k > len(u):
        print(f"k ({k}) is larger than the number of eigenvectors ({len(u)}). Returning x.")
        return x
    elif k == len(u):
        return x
    
    # Keep k largest eigenvectors
    if k >= 0:
        u_subset = u[:, :k]
    else:
        u_subset = u[:, -k:]
        
    P = u_subset @ u_subset.T
    
    # Project x
    x_proj = P @ (x)# - np.mean(x, axis=0))
    #x_proj = x_proj / np.std(x_proj)
    
    return x_proj


def plot_momentumspace_distribution(x: List[float] , ax: plt.Axes = None):
    """
	Plot the activity distribution in momentum space RG.

    :param X: input dataset
    :param ax: plt.Axes object for plotting.
	"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        
    cluster_size = len(x)
    print(f"Cluster size: {cluster_size}")
    
	# Fraction of modes to keep
    modes_list = [2, 16, 32, 64, 128, 256]
	
	# Plot different K with a gradient in color
    alphas = np.logspace(-1, 0, len(modes_list))    
    for i, n_mode in enumerate(modes_list):
        x_proj = np.empty(x.shape)
        print(f"Modes kept: {int(cluster_size / n_mode)}")

        # Compute the correlation matrix
        c = np.cov(x)

        # Compute the eigenspectrum
        eigvals, eigvecs = np.linalg.eigh(c)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Project on subspace
        k = int(cluster_size / n_mode)
        x_proj = projection(x, k, eigvecs)

        # Percentage of variance kept
        variance_proj = 100 * np.sum(eigvals[:k]) / np.sum(eigvals)

        # Plot activity distribution
		#NOTE: bins should not be much larger than 10% of the number of spins
        bins, edges = np.histogram(x_proj.reshape(-1), bins=100, density=True)
        edge_centers = (edges[:-1] + np.roll(edges, -1)[:-1]) / 2
        bin_size = edges[1] - edges[0]

		# Normalize the bins
        bins *= bin_size

        ax.plot(edge_centers, bins, "o--", markersize=3, c=u"#348ABD", alpha=alphas[i], label=f'N/{n_mode}')
        ax.set_yscale("log")
        ax.set_xlabel("Normalized activity", fontsize=30)
        ax.set_ylabel("Probability", fontsize=30)
        ax.legend()


def show_clusters_by_imshow(clusters: List, rg_range: tuple = (0,0)):
    """
    Show the clusters that are formed during the RG procedure.

    Parameters:
        cluster - ndarray containing the clusters at the different iterations of the RG transformation
        verbose (optional) - if True it prints the cluster list after showing the image 
    """

    original_size = len(clusters[0])
    grid = np.zeros((original_size, original_size))

    # Length rg_range
    if rg_range != (0,0):
        clusters = clusters[rg_range[0]:rg_range[1]]
        size = rg_range[1] - rg_range[0]
    else:
        clusters = clusters[1:]
        size = len(clusters)

    # Create figure for images
    fig = plt.figure(figsize=(12, 8))

    # Loop over clusters
    for n, c in enumerate(clusters):
        ax = fig.add_subplot(1, size, n+1)

        # Create grid to show
        colors = np.arange(1, 1 + len(c[:,0])) / len(c[:,0]) * 10
        for i, color in enumerate(colors):
            for j in c[i]:
                grid[j, c[i]] = color
        
        ax.imshow(grid, extent=(0, original_size, 0, original_size), origin="lower")
        ax.set_title(f"Cluster size = {len(c.T)}")

        # Specify the x-axis and y-axis ticks
        ax.set_xticks(range(0, original_size+2, 2))
        ax.set_yticks(range(0, original_size+2, 2))


        # if OUTPUT_DIR != "":
        #     fig.savefig(f"{OUTPUT_DIR}/clusterSize={len(c.T)}")
    return fig

def plot_eigenvalue_scaling(
        X_coarse: List,
        clusters: List,
        rg_range: tuple = (0,0), 
        ax: plt.Axes = None, 
        return_data: bool = False
        ):
    """
    Plot the eigenvalues spectrum of the Pearson correlation matrix at different steps of 
    coarse graining.

    Parameters:
        X_coarse - a list of arrays containing the activity of the orignal and coarse-grained variables. 
    """
    # Create fig, ax
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if rg_range != (0,0):
        X_coarse = X_coarse[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    if rg_range[0] == 0:
        cluster_sizes = [1]
        cluster_sizes += [len(c[0]) for c in clusters[1:]]
    else: 
        cluster_sizes = [len(c[0]) for c in clusters]
        
    alphas = np.logspace(-1, 0, len(X_coarse))
    for i, X in enumerate(X_coarse):
        # Compute correlation matrix
        corr = np.corrcoef(X)

        # Compute its eigenvalues
        eigvalues, eigvectors = np.linalg.eigh(corr)

        # Plot spectrum
        sort_idx = np.argsort(eigvalues)
        eigvalues = eigvalues[sort_idx][::-1]
        rank = np.arange(1, len(eigvalues)+1) / len(eigvalues)
        ax.plot(rank, eigvalues, "o", markersize=3, label=f"K = {cluster_sizes[i]}")#, c=u"#348ABD", alpha=alphas[i])

    # Make plot nice
    ax.set_ylabel("Eigenvalues")
    ax.set_xlabel("Rank/K")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    if return_data == "True":
        return rank, eigvalues


def plot_n_largest_eigenvectors(Xs, n, rg_range=(0,0)):
    """
    Plot the n largest eigenvectors in an imshow figure.

    Parameters:
        X_coarse - a list of arrays containing the activity of the orignal and coarse-grained variables. 
        n - number of eigenvectors to plot
    """
    N = len(Xs[0])
    if rg_range != (0,0):
        Xs = Xs[rg_range[0]:rg_range[1]]

    for j, X in enumerate(Xs):
        corr = np.corrcoef(X)
        eigvalues, eigvectors = np.linalg.eigh(corr)
        eigvalues = eigvalues[::-1]
        eigvectors = eigvectors[:, ::-1]

        plot_size = math.ceil(np.sqrt(n))
        fig, axs = plt.subplots(plot_size, plot_size)
        for i in range(plot_size*plot_size):
            row, col = i // plot_size, i % plot_size
            eigvector = eigvectors[:, i].reshape(-1, 1)
            im = axs[row, col].imshow(eigvector @ eigvector.T)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            fig.subplots_adjust(hspace=.8)
            fig.colorbar(im)

        fig.suptitle(f"Cluster size: {N / len(X)}")
        plt.show()

def plot_eigenvalue_spectra_within_clusters(
        Xs: List, 
        clusters: List, 
        rg_range: tuple = (0,0), 
        ax: plt.Axes = None,
        return_data: bool = False,
        ):
    """
    This function plots the eigenvalue spectra within the clusters. At each coarse-grained level the mean and variance of the spectra
    across the different clusters are computed and plotted.

    Parameters:
        Xs - list contianing the dataset at each coarse-grained level
        clusters - list containing the clusters that where formed at the different coarse-grianing iterations
    """
    original_dataset = Xs[0]

    if rg_range != (0,0):
        if len(rg_range) == 1:
            Xs = Xs[rg_range[0]:]
            clusters = clusters[rg_range[0]:]
        else:
            Xs = Xs[rg_range[0]:rg_range[1]]
            clusters = clusters[rg_range[0]:rg_range[1]]

    # Create figure and ax
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Loop over coarse-graining iterations
    ranks = []
    means = []
    confidence_intervals_l = []
    for i, cluster in enumerate(clusters):

        # Compute cluster size
        try:
            cluster_size = len(cluster[0])
        except TypeError:
            continue
        
        # Not interested in the spectra of these small clusters
        if cluster_size <= 1:
            continue
            
        # Compute the spectrum for each cluster, average and plot with confidence interval
        eigvalues_l = []
        for c in cluster:

            if len(c) != cluster_size:
                continue
            
            corr = np.cov(original_dataset[c])
            eigvalues, _ = np.linalg.eigh(corr)
            eigvalues_l.append(np.sort(eigvalues)[::-1])
         
        # Compute statistics
        rank = np.arange(1, len(eigvalues) + 1) / len(eigvalues)
        mean = np.mean(eigvalues_l, axis=0)
        stds = np.std(eigvalues_l, axis=0)

        confidence_intervals = [
            mean - mean * np.power(10, -1 * stds  / (mean * len(eigvalues_l))),
            mean * np.power(10, stds / (mean * len(eigvalues_l))) - mean,
            ]
        
        # Plot
        ax.errorbar(rank, mean, yerr=confidence_intervals, fmt="^", markersize=5, label=f"K = {cluster_size}")

        # Store 
        ranks.append(rank)
        means.append(mean)
        confidence_intervals_l.append(confidence_intervals)
                
    ax.set_xlabel("Rank/K")
    ax.set_ylabel("Eigenvalues")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    if return_data:
        return ranks, means, confidence_intervals_l


def fit_largest_eigenvalues(
        ranks: List,
        means: List,
        n: int = 2,
        rg_range_len: int = 4,
        label: str = "", 
        fmt: str = ".",
        ax: plt.Axes = None,
        fig_dir: str = None,
):
    """
    Fit the eigenvalue to a power law based on their rank starting from the largest to lowest. 
    A single eigenvalue is taken for each iteration of the PRG procedure.

    :param ranks: list of the ranks for the eigenvalues 
    :param means: list of the mean eigenvalue at each iteration
    :param n: number of eigenvalues to fit
    :param rg_range_len: PRG iterations
    """
    # Store n largest eigenvectors at each iteration in a separate list
    eigenvalues = np.empty(shape=(n, rg_range_len))
    new_ranks = np.empty(shape=(n, rg_range_len))
    for i, mean in enumerate(means):
        eigenvalues[:, i] = mean[:n]
        new_ranks[:, i] = ranks[i][:n]

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5)) 
    
    # Fit power laws to the eigenvalues
    colors = ["C{}".format(i) for i in range(rg_range_len)]

    params = []
    pcovs = []
    for i in range(n):
        row = eigenvalues[i]
        x = new_ranks[i]
        param, pcov = fit_power_law(x, row)
        params.append(param)
        pcovs.append(pcov)
        y_fit = power_law(x, *param)
        
        # Plot the actual values and their power law fit
        if i == 0:
            ax.plot(x, row, fmt, c=colors[i], label=label)
        else:
            ax.plot(x, row, fmt, c=colors[i])
        ax.plot(x, y_fit, '-', c=colors[i], label=r"$\beta_{}$ = {:.2f}".format(i, param[0]))
        
    # Add labels and legend
    ax.set_xlabel('Rank/K')
    ax.set_ylabel('Eigenvalues')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Scaling of largest eigenvalues")
    ax.legend()

    if fig_dir != None:
        filename = Path(fig_dir) / "scaling_{}_largest_eigenvalues".format(n)
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    return params, pcovs


def plot_free_energy_scaling(
    Xs: List[List],
    skip_iters: int = 2, 
    ax: plt.Axes = None,
    label: str = "", 
    color: str = "black",
    show_exponential_decay: bool = True,
):
    """
    Plot the free energy scaling given a list of PRG activity. Skip a few of the first iterations if 'skip_iters' is specified
    and if only a sinle cluster size has a free energy let the user know.

    :param Xs: list of activity
    :param skip_iters: number iterations to skip
    :param ax: Axes to plot on
    :param label: label for the plot
    :param color: color for the plot
    :param show_exponential_decay: if true show scaling of exponential decay (\beta=1)
    """
    # Compute free energy
    free_energies, cluster_sizes = compute_free_energies(Xs, skip_iters)
    if len(free_energies) < 2:
        print("Clusters are only silent at size K = [{}]. Can not fit a power law for {}".format(cluster_sizes, label))
        return -1
    
    # Fit power law
    params, pcov = fit_power_law(cluster_sizes, free_energies)
    free_energies_fit = power_law(cluster_sizes, *params)

    # Plot
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(cluster_sizes, free_energies, "o", color=color, markersize=5, label=label)
    ax.plot(cluster_sizes, free_energies_fit, "-", color=color, label=r"$\beta$ = {:.2f}".format(params[0]))
    if show_exponential_decay:
        ax.plot(cluster_sizes, cluster_sizes, "--", color="gray", label=r"$\beta$ = 1") # Exponential decay
    ax.set_xlabel("Cluster size")
    ax.set_ylabel(r"-ln($P_0$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if label != "":
        ax.legend()

def compute_free_energies(Xs: List[List], skip_iters: int = 2) -> List[np.ndarray]:
    """
    Given a list of PRG activity compute the free energies at each iteration for each cluster. 
    Skip the first 'skip_iters' iterations as these can be noisy.

    :param Xs: input activity
    :param skip_iters: prg iterations to skip

    :return free_energies: list of free energies
    """
    free_energies = []
    cluster_sizes = []
    for i, x in enumerate(Xs[skip_iters:]):
        values, counts = np.unique(x, return_counts=True)
        zero_idx = np.argwhere(values == 0)
        probablity_zero = counts[zero_idx] / np.sum(counts)
        if len(zero_idx.reshape(-1)) != 0:
            free_energy = - np.log(probablity_zero)
            free_energies.append(free_energy.reshape(-1)[0])
            cluster_sizes.append(2**(skip_iters + i))
    return free_energies, cluster_sizes


def scaling_moments_data(
    Xs: List[List],
    clusters: List[List],
    moment: int = 2,
    show_distributions: bool = False,
):
    """
    Compute the data that would be needed for the scaling plots of the moments. These are the mean moment
    of the clusters for at each iteration. The returned values also include a confidence interval.

    :param Xs: list of activity
    :param clusters: list of clusters
    :param moment: moment to compute

    :return cluster_size: x-axis in the moments plot
    :return means: y-axis in the moments plot
    :return confidence_intervals: 3*SME for the means
    :return params: power law fit exponent and base (b,a)
    :return pcov: variance of exponent fit
    """
    means = []
    cluster_sizes = []
    confidence_intervals = np.empty(shape=(len(Xs), 2))

    for i, X in enumerate(Xs):
        cluster_size = len(clusters[0]) / len(clusters[i])
        cluster_sizes.append(cluster_size)
        X = X * cluster_size

        if moment % 2 == 1:
            moments = np.abs(sp_moment(X, moment=moment, axis=1))
        else:
            moments = sp_moment(X, moment=moment, axis=1)
        
        mean = moments.mean()
        means.append(mean)
        
        confidence_intervals[i] = [
            mean - mean * np.power(10, -1 * moments.std() / (mean * np.sqrt(len(moments)))),
            mean * np.power(10, moments.std() / (mean * np.sqrt(len(moments)))) - mean,
            ]

        if show_distributions:
            print(
                mean,
                moments.std(),
                len(moments),
                np.sqrt(len(moments)),
                moments.std() / (mean * np.sqrt(len(moments))),
                np.power(10, -1 * moments.std() / (mean * np.sqrt(len(moments)))),
                np.power(10, moments.std() / (mean * np.sqrt(len(moments)))),
                confidence_intervals[i]
            )
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.plot(range(len(moments)), moments, ".")
            ax.set_ylabel(f"Mean of {moment}-moment")

            ax.axhline(mean - confidence_intervals[i][0], linestyle="--", label="CI lower")
            if (confidence_intervals[i][1] / mean) < 1e3:
                ax.axhline(mean + confidence_intervals[i][1], linestyle="--", label="CI upper")
            ax.legend()
            plt.show()
            
        params, pcov = fit_power_law_fixed_a(cluster_sizes, means)
        params = list(params) + [means[0]] # (b, a)

    return cluster_sizes, means, confidence_intervals, params, pcov

def plot_scaling_of_moment(
    Xs: List[List],
    clusters: List[List],
    moment: int = 2,
    ax: plt.Axes = None,
    label: str = "",
    color: str = "black",
    show_distributions: bool = False,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    (
        cluster_sizes,
        means,
        confidence_intervals,
        params,
        pcov,
    ) = scaling_moments_data(Xs, clusters, moment, show_distributions=show_distributions)
    ax.errorbar(
        cluster_sizes,
        means,
        confidence_intervals.T,
        markersize=5,
        fmt="o",
        color=color,
        label=label
    )

    # Plot zero correlation limit
    limitK1 = params[1] * np.array(cluster_sizes)
    ax.plot(cluster_sizes, limitK1, "--", color="gray", alpha=0.5)

    # Plot maximum correlation limit
    limitK2 = params[1] * np.array(cluster_sizes) ** moment
    ax.plot(cluster_sizes, limitK2, "--", color="gray", alpha=0.5)

    ax.plot(
        cluster_sizes,
        power_law(cluster_sizes, params[0], params[1]), 
        "-", 
        c=color, 
        label=r"$\alpha$ ({}): {:.2f}".format(label, params[0]),
        )

    ax.set_xlabel("Cluster size K")
    ax.set_ylabel(f"{moment} moment")
    ax.set_yscale("log")
    ax.set_xscale("log")
    if label != "":
        ax.legend()
