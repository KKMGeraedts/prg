import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy.stats import binom
from scipy.stats import moment as sp_moment
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import scipy.odr as odr

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

# def fit_power_law2(x, y):
#     # Create a model for fitting.
#     power_law_model = odr.Model(lambda p, x: power_law(x, *p))
#     
#     # Create a RealData object using your initiated data.
#     data = odr.RealData(x, y)
#     
#     # Set up ODR with the model and data.
#     odr_obj = odr.ODR(data, power_law_model, beta0=[y[0], 1.])
#     
#     # Run the regression.
#     out = odr_obj.run()
#     
#     # Use the in-built pprint method to give us results.
#     #out.pprint()
# 
#     # Extract the parameters:
#     return out.beta

def plot_normalized_activity(p_averages, p_confidence_intervals, unique_activity_values, clusters, rg_range=(0,0), title="", binom_fit=False):
    """
    Plots the distribution of the normalized activity. Given the average probabilities, standard deviations and the unique values at each
    step of the coarse-graining. This data can be obtained from the RG_class.

    Parameters:
        p_averages - a list of size = n_rg_iterations with each list containing an numpy array of the average activity across all clusters
        p_confidence_intervals - a list of size = n_rg_iterations with each item containing a list of upper and lower values for 95% confidence interval
        unique_activity_values - a list of size = n_rg_iterations with each list containing an numpy array of the unique activity values in the clusters
    """
    cluster_sizes = [1] + [len(c[0]) for c in clusters[1:]]

    # Create fig, ax
    fig, ax = plt.subplots(1)
    
    if rg_range != (0,0):
        cluster_sizes = cluster_sizes[rg_range[0]:rg_range[1]]
        p_averages = p_averages[rg_range[0]:rg_range[1]]
        p_confidence_intervals = p_confidence_intervals[rg_range[0]:rg_range[1]]
        unique_activity_values = unique_activity_values[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    for i, _ in enumerate(p_averages):
        p_confidence_interval = np.abs(p_confidence_intervals[i].T - p_averages[i])
        ax.errorbar(unique_activity_values[i], p_averages[i], yerr=np.array(p_confidence_interval), fmt="o--", linewidth=1, markersize=3, label=f"K = {cluster_sizes[i]}")
        
        # Binomial distribution
        if binom_fit == True:
            n_trails = cluster_sizes[i]
            p = 0.5
            x = np.arange(binom.ppf(0.01, n_trails, p), binom.ppf(1	, n_trails, p)+1)
            binom_pdf = binom.pmf(x, n_trails, p)

            # Plot binomail 
            ax.plot(x / n_trails, binom_pdf, '--', color="grey", alpha=0.3)
        
    # Make plot nice
    ax.set_ylabel("probability")
    ax.set_xlabel("normalized activity")
    #ax.set_title("Probability distribution of the normalized activity")
    ax.grid(True)
    
    ax.legend(loc="upper right", ncol=2)
    ax.set_yscale("log")
    #ax.set_ylim(0, 0.8)

    return fig, ax

def show_clusters_by_imshow(clusters, rg_range=(0,0)):
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
        ax.set_title(f"cluster size = {len(c.T)}")

        # Specify the x-axis and y-axis ticks
        ax.set_xticks(range(0, original_size+2, 2))
        ax.set_yticks(range(0, original_size+2, 2))


        # if OUTPUT_DIR != "":
        #     fig.savefig(f"{OUTPUT_DIR}/clusterSize={len(c.T)}")
    return fig

def plot_eigenvalue_scaling(X_coarse, clusters, rg_range=(0,0), ax=None):
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
        eigvalues, eigvectors = np.linalg.eig(corr)

        # Check complex part of eigenvalues
        delta = 10e-3
        large_complex_eigvalues = eigvalues.imag[eigvalues.imag > delta]
        if large_complex_eigvalues != []:
            print(f"Found some eigenvalues with complex part larger than {delta}. Ignoring them for now. {large_complex_eigvalues}")

        eigvalues = eigvalues.real

        # Plot spectrum
        sort_idx = np.argsort(eigvalues)
        eigvalues = eigvalues[sort_idx][::-1]
        rank = np.arange(1, len(eigvalues)+1) / len(eigvalues)
        ax.plot(rank, eigvalues, "o", markersize=3, label=f"K = {cluster_sizes[i]}")#, c=u"#348ABD", alpha=alphas[i])

    # Make plot nice
    ax.set_ylabel("eigenvalues")
    ax.set_xlabel("rank/K")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


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

def plot_eigenvalue_spectra_within_clusters(Xs, clusters, rg_range=(0,0)):
    """
    This function plots the eigenvalue spectra within the clusters. At each coarse-grained level the mean and variance of the spectra
    across the different clusters are computed and plotted.

    Parameters:
        Xs - list contianing the dataset at each coarse-grained level
        clusters - list containing the clusters that where formed at the different coarse-grianing iterations
    """
    original_dataset = Xs[0]

    if rg_range != (0,0):
        Xs = Xs[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    # Create figure and ax
    fig, ax = plt.subplots(1)

    # Loop over coarse-graining iterations
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
            
            corr = np.corrcoef(original_dataset[c])
            eigvalues, _ = np.linalg.eig(corr)
            eigvalues_l.append(np.sort(eigvalues)[::-1])
         
        # Compute statistics
        rank = np.arange(1, len(eigvalues) + 1) / len(eigvalues)
        mean = np.mean(eigvalues_l, axis=0)
        std = np.std(eigvalues_l, axis=0)

        # Bootstrap params
        N = 1000
        percentile = 2.5 # =(100-confidence)/2 
        confidence_intervals = np.empty(shape=(len(eigvalues_l[0]), 2))

        # Perform bootstrap for the confidence interval
        for j, eigvs in enumerate(np.transpose(eigvalues_l)):
            bootstrap_values = [np.random.choice(eigvs, size=len(eigvs), replace=True).mean() for i in range(N)]
            confidence_intervals[j] = np.percentile(bootstrap_values, [percentile, 100-percentile])
            confidence_intervals[j] = np.abs(confidence_intervals[j] - mean[j])
    
        # if cluster_size == 32:
        #     print(np.array(eigvalues_l)[:, 1])
        #     print(np.mean(np.array(eigvalues_l)[:, 1]))

        # Plot
        ax.errorbar(rank, mean, yerr=confidence_intervals.T, fmt="o", markersize=3, label=f"K = {cluster_size}")
                
    ax.set_xlabel("rank/K")
    ax.set_ylabel("eigenvalues")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    return fig, ax

def plot_free_energy_scaling(p_averages, p_confidence_intervals, unique_activity_values, clusters, individual_cluster_scaling=True):
    """
    When a RG transformation is exact the free energy does not change. This function compute the free energy at each
    coarse-grained step and log plots the values. We hope to see some scaling with a power law close to 1.

    We can compute the free energy by F = -np.ln(p0) with p0 the probability that a cluster is silent.

    Parameters:
        X_list - nd numpy array containing the variables at different steps of the coarse-graining
    """
    # Data
    p0_avg = []
    p0_confidence_intervals = []
    cluster_sizes = [1]
    cluster_sizes += [len(c[0]) for c in clusters[1:]]
    never_silent_clusters = [] # To keep track at which sizes clusters are never silent
    popped = 0

    # # 100% and 0% correlation limits
    # idx = np.argwhere(unique_activity_values[0] == 0.0)
    # limit100 = list(list(p_averages[0][idx])[0]) * len(unique_activity_values)
    # limit0 = (limit100) ** np.arange(1, len(unique_activity_values)+1)

    # Create fig, ax
    fig, ax = plt.subplots(1, figsize=(8, 7))

    for i, unique_vals in enumerate(unique_activity_values):
        # Find idx at which cluster is silent
        idx = np.argwhere(unique_vals == 0.0)
        # Check it exists
        if len(idx) != 0:
            idx = idx[0]
            
            # Add to list
            p0_avg.append(list(p_averages[i][idx])[0])
            p0_confidence_intervals.append([p_confidence_intervals[i][idx][0][0], p_confidence_intervals[i][idx][0][1]])
        else:
            never_silent_clusters.append(cluster_sizes[i-popped])
            cluster_sizes.pop(i - popped)
            popped += 1

    # print(limit0)
            
    # # Plot limits
    # plt.plot(cluster_sizes, limit0, "--", alpha=0.5, label="0% correlation")
    # plt.plot(cluster_sizes, limit100, "--", alpha=0.5, label="100% correlation")

    # Check that clusters are silent
    if cluster_sizes == []:
        print("Cannot plot free energy. Clusters are never silent.")
        plt.close()
        return
    elif never_silent_clusters != []:
        print(f"Clusters with following sizes are never silent: {never_silent_clusters}")
    
    # Free energy from probability of silence
    p0_avg = -np.log(p0_avg)
    p0_confidence_intervals = -np.log(p0_confidence_intervals)
    
    # Fit power law
    params = fit_power_law(cluster_sizes, p0_avg)
    print(f"Parameters of power law fit for free energy: {params}")
    
    # Plot fit
    ax.plot(cluster_sizes, power_law(cluster_sizes, params[0], params[1]), "--", c="gray", alpha=0.5, label=f"power law fit: $\\alpha$={params[0]:.2f}")

    # Plot the probability of the cluster being silent
    p0_confidence_intervals = np.abs(np.transpose(p0_confidence_intervals) - p0_avg)  
    ax.errorbar(cluster_sizes, p0_avg, yerr=p0_confidence_intervals, color="black", fmt="o", markersize=5)
    ax.set_xlabel("cluster size")
    ax.set_ylabel(r"-log P$_{Silence}$")
    #ax.set_ylim(0, max(p0_avg)+0.1)
    plt.yscale("log")
    plt.xscale("log")
    #plt.ylim(0, 1)
    plt.legend()

    return fig, ax

def plot_scaling_of_moment(X_coarse, clusters, moment=2, limits=True, fit=True, fit_fixed_a=False, ax=None):
    """
    We know that if we add to RV together their variance can be computed by Var(X+Y) = Var(X) + Var(Y) + 2Cov(X, Y). If we can assume Var(x)=Var(Y) then
    adding K uncorrelated RVs we get a scaling of the variance with K^1. On the other hand if the RVs are maximally correlated then one would expect
    a scaling with K^2 (Some assumptions were made here). 
    
    Here we plot the two limits, the scaling in the dataset and return the value a.

    Parameters:
        X_coarse - a list of size n_rg_iterations containing each a ndarray of size (n_variables, n_datapoints)
        clusters - a list of size n_rg_iterations containing the indices of the orignal spins that were clustered 

    Return:
        a - scaling found in the coarse-graining procedure
    """
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    x = []
    y = []
    yerr = []
    # Things to keep track of
    moment_avgs = []
    confidence_intervals = np.empty(shape=(len(X_coarse), 2))
    cluster_sizes = []

    # Loop over RGTs
    for i, X in enumerate(X_coarse):

        # Compute cluster size and save
        cluster_size = len(clusters[0]) / len(clusters[i])
        cluster_sizes.append(cluster_size)

        X = X * cluster_size # Unnormalize the activity

        # Compute moment
        n_moment = sp_moment(X, moment=moment, axis=1) # These are the central moments

        # Compute mean
        moment_avgs.append(n_moment.mean())

        # Compute confidence interval by bootstrap
        N = 1000
        percentile = 2.5 
        bootstrap_values = [np.random.choice(n_moment, size=(len(n_moment))).mean() for _ in range(N)]
        confidence_interval = [np.percentile(bootstrap_values, percentile), np.percentile(bootstrap_values, 100-percentile)]
        confidence_intervals[i] = np.abs(moment_avgs[i] - confidence_interval)
    
    # Plot moments along with error
    ax.errorbar(cluster_sizes, moment_avgs, confidence_intervals.T, markersize=5, fmt="o", color="black", alpha=0.8)

    a = moment_avgs[0] # This is used for the limits
    # Fit power law
    if fit == True:
        if fit_fixed_a == True:
            # Fixed a
            params, pcov = fit_power_law_fixed_a(cluster_sizes, moment_avgs)
            params = list(params) + [moment_avgs[0]] # (b, a)
        else:
            # Varying a
            params, pcov = fit_power_law(cluster_sizes, moment_avgs)
            
        print(f"Parameters of power law fit for {moment} order moment: {params}")
        ax.plot(cluster_sizes, power_law(cluster_sizes, params[0], params[1]), "-", c="black", alpha=0.6, label=f"power law fit: $\\alpha$={params[0]:.2f}")

    # Show limits 
    if limits == True:
        # # Plot K^1 limit (for variance)
        limitK1 = a * np.array(cluster_sizes)
        ax.plot(cluster_sizes, limitK1, "--", color="gray", alpha=0.5)

        # # Plot K^2 limit (for variance)
        limitK2 = a * np.array(cluster_sizes) ** moment
        ax.plot(cluster_sizes, limitK2, "--", color="gray", alpha=0.5)
            
    # Make figure look nice
    ax.set_xlabel("cluster size K")
    ax.set_ylabel("activity variance")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
