import argparse
import numpy as np
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD

def load_example_data(data_model):
    # Common positive real-valued timestamps for all data
    timestamps = np.array([
        [1.1, 2.1, 3.1],
        [4.2, 5.2, 6.2],
        [7.3, 8.3, 9.3],
        [10.4, 11.4, 12.4],
        [13.5, 14.5, 15.5],
        [16.6, 17.6, 18.6],
        [19.7, 20.7, 21.7],
        [22.8, 23.8, 24.8],
        [25.9, 26.9, 27.9],
        [29.0, 30.0, 31.0]
    ])

    if data_model in ['events', 'regular_events']:
        # Explicit example data for 'events' and 'regular_events'
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
    elif data_model == 'measures':
        # Explicit example data for 'measures' with real numbers and negatives
        measurements = np.array([
            [1.1, -2.2, 3.3],
            [-4.4, 5.5, -6.6],
            [7.7, -8.8, 9.9],
            [-1.1, 2.2, -3.3],
            [4.4, -5.5, 6.6],
            [-7.7, 8.8, -9.9],
            [1.1, -2.2, 3.3],
            [-4.4, 5.5, -6.6],
            [7.7, -8.8, 9.9],
            [-1.1, 2.2, -3.3]
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
    parser = argparse.ArgumentParser(description='Bayesian Blocks and Matrix Decomposition')
    parser.add_argument('--data_model', type=str, required=True, choices=['events', 'regular_events', 'measures'], help='Type of data model to use')
    parser.add_argument('--n_components', type=int, required=True, help='Number of components for NMF/SVD')
    parser.add_argument('--n_iter', type=int, default=5, help='Number of iterations for the randomized SVD solver')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for NMF')
    parser.add_argument('--p0', type=float, default=0.05, help='False alarm probability for Bayesian Blocks')

    args = parser.parse_args()

    # Load example data based on the chosen data model
    timestamps, measurements = load_example_data(args.data_model)

    # Process data and ensure correct format for 'events' and 'regular_events'
    if args.data_model in ['events', 'regular_events']:
        measurements = np.abs(np.ceil(measurements)).astype(int)  # Ensure positive integers

    # Add jitter to timestamps to ensure uniqueness
    timestamps = add_jitter(timestamps)

    # Flatten the timestamps and measurements for Bayesian Blocks input
    flattened_timestamps = timestamps.flatten()
    flattened_measurements = measurements.flatten()

    # Apply Bayesian Blocks with the correct fitness model
    if args.data_model in ['events', 'regular_events']:
        edges = bayesian_blocks(t=flattened_timestamps, x=flattened_measurements, p0=args.p0, fitness='events')
    elif args.data_model == 'measures':
        edges = bayesian_blocks(t=flattened_timestamps, x=flattened_measurements, p0=args.p0, fitness='measures')

    # Create the data matrix for decomposition
    data_matrix = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        mask = (flattened_timestamps >= start) & (flattened_timestamps < end)
        block_measurements = flattened_measurements[mask]
        data_matrix.append(np.sum(block_measurements))

    # Convert data_matrix to numpy array
    data_matrix = np.array(data_matrix)

    # Check if we need to transpose the matrix
    if data_matrix.ndim == 1:
        data_matrix = data_matrix.reshape(-1, 1)

    # Print the shape of the data matrix
    print(f"Shape of the data matrix: {data_matrix.shape}")

    # Ensure the data matrix has at least 2 columns for TruncatedSVD
    if data_matrix.shape[1] < 2:
        print("Error: Data matrix has less than 2 features, TruncatedSVD requires at least 2 features.")
        return

    # Apply matrix decomposition
    if args.data_model == 'measures':
        decomposer = TruncatedSVD(n_components=args.n_components, n_iter=args.n_iter)
    else:
        decomposer = NMF(n_components=args.n_components, max_iter=args.max_iter)

    W = decomposer.fit_transform(data_matrix)
    H = decomposer.components_

    # Reconstruct the matrix
    reconstructed_data = np.dot(W, H)

    # Plotting the results
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