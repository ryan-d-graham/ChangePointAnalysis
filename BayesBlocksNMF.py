import numpy as np
from astropy.stats import bayesian_blocks as bb
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# Example multivariate event data
# List of timestamps for each variable
timestamps = [
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],  # Variable 1
    [0.7, 1.7, 2.7, 3.7, 4.7, 5.7],  # Variable 2
    [0.8, 1.8, 2.8, 3.8, 4.8, 5.8]   # Variable 3
]

# List of weights corresponding to the timestamps
weights = [
    [1, 2, 1, 2, 1, 2],  # Variable 1
    [2, 1, 2, 1, 2, 1],  # Variable 2
    [1, 1, 1, 1, 1, 1]   # Variable 3
]

# Concatenate the timestamps and weights
combined_timestamps = np.concatenate(timestamps)
combined_weights = np.concatenate(weights)

# Apply Bayesian Blocks to find common change points for all variables
edges = bb(t=combined_timestamps, x=combined_weights, p0=0.05, fitness='events')

# Create a matrix to store the weighted rates for each block
num_variables = len(timestamps)
num_blocks = len(edges) - 1
V = np.zeros((num_variables, num_blocks))

# Calculate the weighted rate for each variable in each block
for i in range(num_variables):
    for j in range(num_blocks):
        start, end = edges[j], edges[j + 1]
        mask = (timestamps[i] >= start) & (timestamps[i] < end)
        if np.sum(mask) > 0:
            V[i, j] = np.sum(weights[i][mask]) / (end - start)

# Apply Non-Negative Matrix Factorization (NMF)
model = NMF(n_components=3, init='random', random_state=0)
W = model.fit_transform(V)
H = model.components_

# Display results
print("Common Change Points:\n", edges)
print("Weighted Rates Matrix V:\n", V)
print("NMF Basis Matrix W:\n", W)
print("NMF Coefficient Matrix H:\n", H)

# Visualization using heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original matrix V
im1 = axes[0].imshow(V, aspect='auto', cmap='inferno')
axes[0].set_title('Input to NMF: Original Matrix V')
axes[0].set_xlabel('Blocks')
axes[0].set_ylabel('Variables')
fig.colorbar(im1, ax=axes[0], orientation='vertical')

# Basis matrix W
im2 = axes[1].imshow(W, aspect='auto', cmap='inferno')
axes[1].set_title('Output of NMF: Basis Matrix W')
axes[1].set_xlabel('Latent Features')
axes[1].set_ylabel('Variables')
fig.colorbar(im2, ax=axes[1], orientation='vertical')

# Coefficient matrix H
im3 = axes[2].imshow(H, aspect='auto', cmap='inferno')
axes[2].set_title('Output of NMF: Coefficient Matrix H')
axes[2].set_xlabel('Blocks')
axes[2].set_ylabel('Latent Features')
fig.colorbar(im3, ax=axes[2], orientation='vertical')

plt.tight_layout()
plt.show()