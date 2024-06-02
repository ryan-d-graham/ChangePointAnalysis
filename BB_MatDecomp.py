import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from astropy.stats import bayesian_blocks as bb

# Example multivariate event data (same as before)


timestamps = [
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],  # Variable 1
    [0.7, 1.7, 2.7, 3.7, 4.7, 5.7],  # Variable 2
    [0.8, 1.8, 2.8, 3.8, 4.8, 5.8]   # Variable 3
] 


weights = [
    [1, 2, 1, 2, 1, 2],  # Variable 1
    [2, 1, 2, 1, 2, 1],  # Variable 2
    [1, 1, 1, 1, 1, 1]   # Variable 3
] 


print("Timestamps shape: ", np.shape(timestamps), "\n")
print("Weights shape: ", np.shape(weights), "\n")

edges = bb(t=combined_timestamps, x=combined_weights, p0=0.05, fitness='events')

num_variables = len(timestamps)
num_blocks = len(edges) - 1
V = np.zeros((num_variables, num_blocks))
for i in range(num_variables):
    for j in range(num_blocks):
        start, end = edges[j], edges[j + 1]
        mask = (timestamps[i] >= start) & (timestamps[i] < end)
        mask_indices = np.where(mask)[0]
        duration = end - start
        if len(mask_indices) > 0 and duration > 0:
            V[i, j] = np.sum(weights[i][mask_indices]) / duration
        else:
            V[i, j] = 0

model = NMF(n_components=3, init='random', random_state=0, max_iter=1000)
W = model.fit_transform(V)
H = model.components_
V_star = np.dot(W, H)

# Visualization using seaborn heatmaps with borders and integer axis ticks
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original matrix V
sns.heatmap(V, ax=axes[0, 0], cmap='inferno', linewidths=.5, linecolor='black', cbar=True, 
            xticklabels=np.arange(1, V.shape[1] + 1), yticklabels=np.arange(1, V.shape[0] + 1))
axes[0, 0].set_title('Input to NMF: Original Matrix V')
axes[0, 0].set_ylabel('Variables')
axes[0, 0].set_xlabel('Blocks')

# Basis matrix W
sns.heatmap(W, ax=axes[0, 1], cmap='inferno', linewidths=.5, linecolor='black', cbar=True, 
            xticklabels=np.arange(1, W.shape[1] + 1), yticklabels=np.arange(1, W.shape[0] + 1))
axes[0, 1].set_title('Output of NMF: Basis Matrix W')
axes[0, 1].set_ylabel('Variables')
axes[0, 1].set_xlabel('Latent Features')

# Coefficient matrix H
sns.heatmap(H, ax=axes[1, 0], cmap='inferno', linewidths=.5, linecolor='black', cbar=True, 
            xticklabels=np.arange(1, H.shape[1] + 1), yticklabels=np.arange(1, H.shape[0] + 1))
axes[1, 0].set_title('Output of NMF: Coefficient Matrix H')
axes[1, 0].set_ylabel('Latent Features')
axes[1, 0].set_xlabel('Blocks')

# Reconstructed matrix V*
sns.heatmap(V_star, ax=axes[1, 1], cmap='inferno', linewidths=.5, linecolor='black', cbar=True, 
            xticklabels=np.arange(1, V_star.shape[1] + 1), yticklabels=np.arange(1, V_star.shape[0] + 1))
axes[1, 1].set_title('Reconstructed Matrix V*')
axes[1, 1].set_ylabel('Variables')
axes[1, 1].set_xlabel('Blocks')

plt.tight_layout()
plt.show()
