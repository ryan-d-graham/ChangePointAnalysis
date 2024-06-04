import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import pandas as pd

# Example data: timestamps and measurements
timestamps = np.array([
    [0.1, 0.15, 0.12],
    [0.21, 0.27, 0.25],
    [0.33, 0.31, 0.38],
    [0.41, 0.45, 0.47],
    [0.53, 0.52, 0.56]
])
measurements = np.array([
    [10, 15, 12],
    [12, 17, 14],
    [11, 16, 13],
    [13, 18, 15],
    [14, 19, 16]
])

# Simulate the Bayesian Blocks result by manually setting edges (simplified example)
edges = np.array([0, 0.2, 0.4, 0.6])

# Create a matrix to store the weighted rates for each block
num_variables = measurements.shape[1]
num_blocks = len(edges) - 1
V = np.zeros((num_variables, num_blocks))

# Calculate the weighted rate for each variable in each block
for i in range(num_variables):
    for j in range(num_blocks):
        start, end = edges[j], edges[j + 1]
        mask = (timestamps[:, i] >= start) & (timestamps[:, i] < end)
        duration = end - start
        if np.sum(mask) > 0 and duration > 0:
            V[i, j] = np.sum(measurements[mask, i]) / duration
        else:
            V[i, j] = 0  # Handle zero counts or zero duration safely

# Standardize the matrix
scaler = StandardScaler()
V_scaled = scaler.fit_transform(V.T)

# Apply Kernel-PCA with Gaussian (RBF) kernel
kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1, fit_inverse_transform=True)
V_transformed = kpca.fit_transform(V_scaled)

# Apply Lasso for sparsity
lasso = Lasso(alpha=0.1)
lasso.fit(V_transformed, np.zeros(V_transformed.shape[0]))
V_sparse = lasso.coef_.reshape(1, -1) * V_transformed

# Visualization using Seaborn
sns.set(style="whitegrid")

# Heatmap of the sparse kernel PCA components
plt.figure(figsize=(10, 8))
sns.heatmap(V_sparse, cmap='viridis', annot=True, linewidths=.5)
plt.title('Heatmap of Sparse Kernel-PCA Transformed Data')
plt.xlabel('Components')
plt.ylabel('Samples')
plt.show()

# Pairplot of the sparse kernel PCA components
sparse_df = pd.DataFrame(V_sparse, columns=[f'Component {i+1}' for i in range(V_sparse.shape[1])])
sns.pairplot(sparse_df)
plt.suptitle('Pairplot of Sparse Kernel-PCA Transformed Data', y=1.02)
plt.show()
