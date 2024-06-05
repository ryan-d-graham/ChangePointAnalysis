import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF

# Example data: unique timestamps and measurements
np.random.seed(42)
timestamps = np.sort(np.random.uniform(0, 1, (100, 10)), axis=0)

# Generate a matrix of lambdas from a gamma distribution
shape, scale = 2., 2.  # shape and scale parameters for gamma distribution
lambdas = np.random.gamma(shape, scale, (100, 10))

# Generate measurements using the lambda matrix for Poisson random draws
measurements = np.random.poisson(lam=lambdas)

# Normalize the matrix to ensure all values are non-negative
scaler = MinMaxScaler()
V_scaled = scaler.fit_transform(measurements)  # (variables x blocks)

# Transpose V_scaled to get the correct shape for NMF
V_scaled_T = V_scaled.T  # (blocks x variables)

# Construct the similarity matrix using k-nearest neighbors
knn_graph = kneighbors_graph(V_scaled_T, n_neighbors=5, mode='connectivity', include_self=True)
S = knn_graph.toarray()

# Compute the degree matrix
D = np.diag(np.sum(S, axis=1))

# Compute the graph Laplacian
L = D - S

# Regularization parameter
alpha = 0.1

# Initialize W and H using NMF
nmf = NMF(n_components=3, init='random', random_state=42)
H = nmf.fit_transform(V_scaled_T)  # (blocks x components)
W = nmf.components_  # (components x variables)

# Transpose W to get the correct shape (variables x components)
W = W.T  # (variables x components)

# Iteratively update W and H
max_iter = 200
for _ in range(max_iter):
    # Update H
    A = np.dot(W.T, W) + alpha * np.identity(W.shape[1])
    B = np.dot(W.T, V_scaled)
    H = np.linalg.solve(A, B.T).T
    H[H < 0] = 0

    # Update W
    A = np.dot(H, H.T)
    B = np.dot(V_scaled_T.T, H.T)
    W = np.linalg.solve(A.T, B.T).T
    W[W < 0] = 0

# Visualization using Seaborn
sns.set(style="whitegrid")

# Heatmap of the GNMF components H
plt.figure(figsize=(10, 8))
sns.heatmap(H, cmap='viridis', annot=True, linewidths=.5)
plt.title('Heatmap of GNMF Components H')
plt.xlabel('Blocks')
plt.ylabel('Components')
plt.show()

# Heatmap of the basis matrix W
plt.figure(figsize=(10, 8))
sns.heatmap(W, cmap='viridis', annot=True, linewidths=.5)
plt.title('Heatmap of GNMF Basis Matrix W')
plt.xlabel('Components')
plt.ylabel('Variables')
plt.show()

# Heatmap of the graph Laplacian L
plt.figure(figsize=(10, 8))
sns.heatmap(L, cmap='viridis', annot=True, linewidths=.5)
plt.title('Heatmap of Graph Laplacian L')
plt.xlabel('Nodes (Blocks)')
plt.ylabel('Nodes (Blocks)')
plt.show()