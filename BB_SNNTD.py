import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_tucker
from astropy.stats import bayesian_blocks
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bayesian Blocks and SNNTD Analysis')
    parser.add_argument('--nntd_rank', type=str, default='10,5,5', help='Comma-separated ranks for the SNNTD')
    parser.add_argument('--p0', type=float, default=0.05, help='False positive rate for Bayesian Blocks')
    return parser.parse_args()

def load_example_data():
    # Generate synthetic MEA data with inhomogeneous timestamps
    np.random.seed(42)
    epsilon = 1e-9
    n_obs = 10
    T = 10
    mea_rows, mea_cols = 8, 8
    mea_channels = mea_rows * mea_cols
    timestamps = [np.sort(np.random.uniform(epsilon, T, n_obs)) for _ in range(mea_rows*mea_cols)]

    # Generate lambda values from a gamma distribution
    shape, scale = 2.0, 1.0
    lambdas = np.random.gamma(shape, scale, (n_obs, mea_channels))

    # Generate Poisson-distributed data using the lambda matrix and ensure positive integers
    measurements = np.random.poisson(lam=lambdas) + 1  # Shift by 1 to ensure all values are positive integers

    return timestamps, measurements

def main():
    args = parse_arguments()
    ranks = tuple(map(int, args.nntd_rank.split(',')))

    # Load example data
    timestamps, measurements = load_example_data()

    # Flatten the data for Bayesian Blocks input
    time_tags = np.concatenate(timestamps)
    measurements_flat = measurements.flatten()

    # Apply Bayesian Blocks to find change points
    edges = bayesian_blocks(t=time_tags, x=measurements_flat, fitness='events', p0=args.p0)
    print("Detected change points:", edges)

    # Create a 3D tensor to store the weighted rates for each block
    num_blocks = len(edges) - 1
    mea_rows, mea_cols = 8, 8
    V = np.zeros((num_blocks, mea_rows, mea_cols))

    # Calculate the weighted rate for each channel in each block
    for i in range(mea_channels):
        row, col = divmod(i, mea_cols)
        for j in range(num_blocks):
            start, end = edges[j], edges[j + 1]
            mask = (timestamps[i] >= start) & (timestamps[i] < end)
            duration = end - start
            if np.sum(mask) > 0 and duration > 0:
                V[j, row, col] = np.sum(measurements[mask, i]) / duration
            else:
                V[j, row, col] = 0  # Handle zero counts or zero duration safely

    # Convert to a tensorly tensor
    V_tensor = tl.tensor(V, dtype=tl.float32)

    # Perform Non-Negative Tucker Decomposition
    core, factors = non_negative_tucker(V_tensor, rank=ranks, n_iter_max=100, tol=1e-5)

    # Function to enforce sparsity
    def enforce_sparsity(tensor, threshold):
        return tl.where(tensor < threshold, 0, tensor)

    # Apply sparsity constraint
    sparsity_threshold = 0.1
    core = enforce_sparsity(core, sparsity_threshold)
    factors = [enforce_sparsity(factor, sparsity_threshold) for factor in factors]

    # Reconstruct the tensor from the decomposition
    V_reconstructed = tl.tucker_to_tensor((core, factors))

    # Plot factor matrices
    def plot_factor_matrix(matrix, title, xlabel, ylabel):
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    plot_factor_matrix(factors[0], "Temporal Factor Matrix (A)", "Latent Factors", "Time Segments")
    plot_factor_matrix(factors[1], "Row Factor Matrix (B)", "Latent Factors", "MEA Rows")
    plot_factor_matrix(factors[2], "Column Factor Matrix (C)", "Latent Factors", "MEA Columns")

    # Plot core tensor slices
    def plot_core_tensor_slices(core, title):
        fig, axes = plt.subplots(1, core.shape[0], figsize=(15, 5))
        for i in range(core.shape[0]):
            ax = axes[i]
            im = ax.imshow(core[i, :, :], cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Slice {i+1}')
        plt.suptitle(title)
        plt.show()

    plot_core_tensor_slices(core, "Core Tensor Slices")

    # Plot original tensor slices
    def plot_tensor_slices(tensor, title):
        num_slices = tensor.shape[0]
        fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
        for i in range(num_slices):
            ax = axes[i]
            im = ax.imshow(tensor[i, :, :], cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Slice {i+1}')
        plt.suptitle(title)
        plt.show()

    plot_tensor_slices(V, "Original Tensor Slices")
    plot_tensor_slices(V_reconstructed, "Reconstructed Tensor Slices")

if __name__ == "__main__":
    main()