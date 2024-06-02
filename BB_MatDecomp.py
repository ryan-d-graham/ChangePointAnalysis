import argparse
import numpy as np
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

def load_example_data():
    timestamps = np.array([
        [1.1, 2.1, 3.1],
        [4.2, 4.2, 6.2],
        [7.3, 8.3, 9.3],
        [10.4, 11.4, 12.4],
        [13.5, 14.5, 15.5],
        [16.6, 17.6, 18.6],
        [19.7, 20.7, 21.7],
        [22.8, 23.8, 24.8],
        [25.9, 26.9, 27.9],
        [29.0, 30.0, 31.0]
    ])
    
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

    timestamps, measurements = load_example_data()

    timestamps = add_jitter(timestamps)

    data_matrix = []
    for i in range(measurements.shape[1]):
        flattened_timestamps = timestamps[:, i]
        flattened_measurements = measurements[:, i]
        
        edges = bayesian_blocks(t=flattened_timestamps, x=flattened_measurements, p0=args.p0, fitness='events')

        column_data = []
        for j in range(len(edges) - 1):
            start, end = edges[j], edges[j + 1]
            mask = (flattened_timestamps >= start) & (flattened_timestamps < end)
            block_measurements = flattened_measurements[mask]
            column_data.append(np.sum(block_measurements))
        data_matrix.append(column_data)

    data_matrix = np.array(data_matrix).T

    print(f"Shape of the data matrix: {data_matrix.shape}")

    if data_matrix.shape[1] < 2:
        print("Error: Data matrix has less than 2 features, NMF requires at least 2 features.")
        return

    decomposer = NMF(n_components=args.nmf_components, max_iter=args.max_iter)
    W = decomposer.fit_transform(data_matrix)
    H = decomposer.components_

    reconstructed_data = np.dot(W, H)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.title('Original Data Matrix')
    plt.imshow(data_matrix, aspect='auto', cmap='inferno')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.title('Reconstructed Data Matrix')
    plt.imshow(reconstructed_data, aspect='auto', cmap='inferno')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.title('Basis Matrix W')
    plt.imshow(W, aspect='auto', cmap='inferno')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()