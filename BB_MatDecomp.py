import argparse
import numpy as np
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD

def load_example_data(data_model):
    if data_model in ['events', 'regular_events']:
        # Generate example data with positive integers for 'events' and 'regular_events'
        timestamps = np.cumsum(np.random.poisson(5, size=(100, 3)), axis=0)
        measurements = np.random.randint(1, 10, size=(100, 3))  # Ensure positive integers
    elif data_model == 'measures':
        # Generate example data with real numbers for 'measures'
        timestamps = np.cumsum(np.random.uniform(0, 5, size=(100, 3)), axis=0)
        measurements = np.random.randn(100, 3)  # Real numbers, can include negatives
    return timestamps, measurements

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

    # Flatten the timestamps and measurements for Bayesian Blocks input
    flattened_timestamps = timestamps.flatten()
    flattened_measurements = measurements.flatten()

    # Apply Bayesian Blocks
    edges = bayesian_blocks(t=flattened_timestamps, x=flattened_measurements, p0=args.p0)

    # Create the data matrix for decomposition
    data_matrix = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        mask = (flattened_timestamps >= start) & (flattened_timestamps < end)
        block_measurements = flattened_measurements[mask]
        data_matrix.append(np.sum(block_measurements))

    data_matrix = np.array(data_matrix).reshape(-1, len(edges) - 1)

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