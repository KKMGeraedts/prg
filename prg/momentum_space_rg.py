from typing import List

import numpy as np
import matplotlib.pyplot as plt

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

def activity_distribution(x: List, ax: plt.Axes = None):
	"""
	Plot the activity distribution in momentum space RG.
	"""
	
	if ax == None:	
		fig, ax = plt.subplots(1, 1, figsize=(7, 6))
	
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

