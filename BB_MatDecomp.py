import argparse
import numpy as np
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import seaborn as sns

def load_example_data():
    # Common positive real-valued timestamps for all data
    timestamps = np.array([
        [1.1, 2.1, 3.1],
        [4.2, 4.2, 6.2],  # Duplicate timestamp for testing jitter
        [7.3, 8.3, 9.3],
        [10.4, 11.4, 12.4],
        [13.5, 14.5, 15.5],
        [16.6, 17.6, 18.6],
        [19.7, 20.7, 21.7],
        [22.8, 23.8, 24.8],
        [25.9, 26.9, 27.9],
        [29.0, 30.0, 31.0]
    ])

    # Explicit example data for 'events'
    measurements = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11, 12]
    ])
    
    return timestamps, measurements

def add_jitter(timestamps):
    """Add a small jitter to repeated timestamps to ensure uniqueness."""
    jitter = 1e-9
    for i in range(timestamps.shape[1]):
        unique_timestamps, counts = np.unique(timestamps[:, i], return_counts=True)
        duplicates = unique_timestamps[counts > 1]
        for dup in duplicates:
            dup_indices = np.where(timestamps[:, i] == dup)[0]
            timestamps[dup_indices, i] += np.linspace(0, jitter * len(dup_indices), len(dup_indices))
    return timestamps

def main():
    parser = argparse.ArgumentParser(description='Bayesian Blocks and Non-Negative Matrix Factorization (NMF)')
    parser.add_argument('--nmf_components', type=int, required=True, help='Number of components for NMF')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for NMF')
    parser.add_argument('--p0', type=float, default=0.05, help='False alarm probability for Bayesian Blocks')
    
    args = parser.parse_args()

    # Load example data
    timestamps, measurements = load_example_data()

    # Add jitter to timestamps to ensure uniqueness
    timestamps = add_jitter(timestamps)

    # Combine all variables' timestamps for common time breaks
    combined_timestamps = timestamps.flatten()
    combined_measurements = measurements.flatten()

    # Sort the combined arrays to maintain order
    sorted_indices = np.argsort(combined_timestamps)
    combined_timestamps = combined_timestamps[sorted_indices]
    combined_measurements = combined_measurements[sorted_indices]

    # Apply Bayesian Blocks on the combined data
    edges = bayesian_blocks(t=combined_timestamps, x=combined_measurements, p0=args.p0, fitness='events')

    # Create the data matrix for decomposition
    num_variables = measurements.shape[1]
    num_blocks = len(edges) - 1
    data_matrix = np.zeros((num_blocks, num_variables))

    for i in range(num_variables):
        flattened_timestamps = timestamps[:, i]
        flattened_measurements = measurements[:, i]

        for j in range(num_blocks):
            start, end = edges[j], edges[j + 1]
            mask = (flattened_timestamps >= start) & (flattened_timestamps < end)
            data_matrix[j, i] = np.sum(flattened_measurements[mask])

    # Print the shape of the data matrix
    print(f"Shape of the data matrix: {data_matrix.shape}")

    # Ensure the data matrix has at least 2 columns for NMF
    if data_matrix.shape[1] < 2:
        print("Error: Data matrix has less than 2 features, NMF requires at least 2 features.")
        return

    # Apply NMF
    decomposer = NMF(n_components=args.nmf_components, max_iter=args.max_iter)
    W = decomposer.fit_transform(data_matrix)
    H = decomposer.components_

    # Reconstruct the matrix
    reconstructed_data = np.dot(W, H)

    # Plotting the results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.title('Original Data Matrix')
    sns.heatmap(data_matrix, cmap='inferno', linewidths=.5, linecolor='white')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Reconstructed Data Matrix')
    sns.heatmap(reconstructed_data, cmap='inferno', linewidths=.5, linecolor='white')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('Basis Matrix W')
    sns.heatmap(W, cmap='inferno', linewidths=.5, linecolor='white')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('Coefficient Matrix H')
    sns.heatmap(H, cmap='inferno', linewidths=.5, linecolor='white')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()