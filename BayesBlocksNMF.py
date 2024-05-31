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
    [1, 2, 8, 2, 10, 1],  # Variable 1
    [2, 7, 2, 3, 1, 5],  # Variable 2
    [1, 10, 1, 15, 1, 4]   # Variable 3
]

# Convert lists to numpy arrays for easier manipulation
timestamps = [np.array(ts) for ts in timestamps]
weights = [np.array(wt) for wt in weights]

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
        mask_indices = np.where(mask)[0]
        duration = end - start
        if len(mask_indices) > 0 and duration > 0:
            V[i, j] = np.sum(weights[i][mask_indices]) / duration
        else:
            V[i, j] = 0  # Handle zero counts or zero duration safely

# Apply Non-Negative Matrix Factorization (NMF)
model = NMF(n_components=3, init='random', random_state=0)
W = model.fit_transform(V)
H = model.components_

# Reconstruct V* using W and H
V_star = np.dot(W, H)

# Display results
print("Common Change Points:\n", edges)
print("Weighted Rates Matrix V:\n", V)
print("NMF Basis Matrix W:\n", W)
print("NMF Coefficient Matrix H:\n", H)
print("Reconstructed Matrix V*:\n", V_star)

# Visualization using heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original matrix V
im1 = axes[0, 0].imshow(V, aspect='auto', cmap='inferno')
axes[0, 0].set_title('Input to NMF: Original Matrix V')
axes[0, 0].set_xlabel('Blocks')
axes[0, 0].set_ylabel('Variables')
fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

# Basis matrix W
im2 = axes[0, 1].imshow(W, aspect='auto', cmap='inferno')
axes[0, 1].set_title('Output of NMF: Basis Matrix W')
axes[0, 1].set_xlabel('Latent Features')
axes[0, 1].set_ylabel('Variables')
fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

# Coefficient matrix H
im3 = axes[1, 0].imshow(H, aspect='auto', cmap='inferno')
axes[1, 0].set_title('Output of NMF: Coefficient Matrix H')
axes[1, 0].set_xlabel('Blocks')
axes[1, 0].set_ylabel('Latent Features')
fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')

# Reconstructed matrix V*
im4 = axes[1, 1].imshow(V_star, aspect='auto', cmap='inferno')
axes[1, 1].set_title('Reconstructed Matrix V*')
axes[1, 1].set_xlabel('Blocks')
axes[1, 1].set_ylabel('Variables')
fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')

plt.tight_layout()
plt.show()
