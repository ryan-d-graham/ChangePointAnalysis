import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
import pandas as pd
from astropy.stats import bayesian_blocks

# Example data: unique timestamps and measurements
np.random.seed(42)
timestamps = np.sort(np.random.uniform(0, 1, (100, 10)), axis=0)

# Generate a matrix of lambdas from a gamma distribution
shape, scale = 2., 2.  # shape and scale parameters for gamma distribution
lambdas = np.random.gamma(shape, scale, (100, 10))

# Generate measurements using the lambda matrix for Poisson random draws
measurements = np.random.poisson(lam=lambdas)

# Flatten the data for Bayesian Blocks input
time_tags = timestamps.flatten()
measurements_flat = measurements.flatten()

# Apply Bayesian Blocks to find change points
edges = bayesian_blocks(t=time_tags, x=measurements_flat, fitness='events', p0=0.05)
print("Detected change points:", edges)

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

# Use LassoCV to find the optimal alpha through cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5, max_iter=10000, tol=1e-4)
lasso_cv.fit(V_transformed, np.zeros(V_transformed.shape[0]))

# Extract the best alpha
best_alpha = lasso_cv.alpha_
print("Best alpha found by LassoCV:", best_alpha)

# Fit Lasso with the best alpha and a different solver
lasso = Lasso(alpha=best_alpha, max_iter=20000, tol=1e-3, selection='random')
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