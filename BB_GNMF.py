import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csgraph
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

# Standardize the matrix
scaler = StandardScaler()
V_scaled = scaler.fit_transform(measurements.T)

# Construct the similarity matrix using k-nearest neighbors
knn_graph = kneighbors_graph(V_scaled, n_neighbors=5, mode='connectivity', include_self=True)
S = knn_graph.toarray()

# Compute the degree matrix
D = np.diag(np.sum(S, axis=1))

# Compute the graph Laplacian
L = D - S

# Regularization parameter
alpha = 0.1

# Initialize W and H
nmf = NMF(n_components=3, init='random', random_state=42)
W = nmf.fit_transform(V_scaled)
H = nmf.components_

# Iteratively update W and H
max_iter = 200
for _ in range(max_iter):
    H = np.linalg.solve(np.dot(W.T, W) + alpha * L, np.dot(W.T, V_scaled))
    H[H < 0] = 0
    W = np.linalg.solve(np.dot(H, H.T), np.dot(V_scaled, H.T)).T
    W[W < 0] = 0

# Visualization using Seaborn
sns.set(style="whitegrid")

# Heatmap of the GNMF components
plt.figure(figsize=(10, 8))
sns.heatmap(H, cmap='viridis', annot=True, linewidths=.5)
plt.title('Heatmap of GNMF Components')
plt.xlabel('Components')
plt.ylabel('Features')
plt.show()
