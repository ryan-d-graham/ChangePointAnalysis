import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_tucker
from astropy.stats import bayesian_blocks
import tensorly as tl
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bayesian Blocks and SNNTD Analysis')
    parser.add_argument('--nntd_rank', type=str, default='10,5,5', help='Comma-separated ranks for the SNNTD')
    parser.add_argument('--p0', type=float, default=0.05, help='False positive rate for Bayesian Blocks')
    parser.add_argument('--epsilon', type=float, default=1e-10, help='Epsilon value to avoid zero timestamps')
    parser.add_argument('--mea_rows', type=int, default=8, help='Number of rows in MEA grid')
    parser.add_argument('--mea_cols', type=int, default=8, help='Number of columns in MEA grid')
    parser.add_argument('--sparsity', type=float, default=0.1, help='Sparsity threshold for enforcing sparsity in decomposition')
    return parser.parse_args()

def load_example_data(epsilon, mea_channels):
    # Generate synthetic MEA data with inhomogeneous timestamps
    np.random.seed(42)
    timestamps = [np.sort(np.random.uniform(epsilon, 10, 100)) for _ in range(mea_channels)]

    # Generate lambda values from a gamma distribution
    shape, scale = 2.0, 1.0
    lambdas = np.random.gamma(shape, scale, (100, mea_channels))

    # Generate Poisson-distributed data using the lambda matrix and ensure positive integers
    measurements = np.random.poisson(lam=lambdas) + 1  # Shift by 1 to ensure all values are positive integers

    return timestamps, measurements

def main():
    args = parse_arguments()
    ranks = tuple(map(int, args.nntd_rank.split(',')))

    # Load example data
    mea_channels = args.mea_rows * args.mea_cols
    timestamps, measurements = load_example_data(args.epsilon, mea_channels)

    # Flatten the data for Bayesian Blocks input
    time_tags = np.concatenate(timestamps)
    measurements_flat = measurements.flatten()

    # Apply Bayesian Blocks to find change points
    edges = bayesian_blocks(t=time_tags, x=measurements_flat, fitness='events', p0=args.p0)
    print("Detected change points:", edges)

    # Create a 3D tensor to store the weighted rates for each block
    num_blocks = len(edges) - 1
    mea_rows, mea_cols = args.mea_rows, args.mea_cols
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

    # Perform Non-Negative Tucker Decomposition with proper initialization
    core, factors = non_negative_tucker(V_tensor, rank=ranks, n_iter_max=100, tol=1e-5, init='random')

    # Function to enforce sparsity
    def enforce_sparsity(tensor, threshold):
        return tl.where(tensor < threshold, 0, tensor)

    # Apply sparsity constraint
    sparsity_threshold = args.sparsity
    core = enforce_sparsity(core, sparsity_threshold)
    factors = [enforce_sparsity(factor, sparsity_threshold) for factor in factors]

    # Reconstruct the tensor from the decomposition
    V_reconstructed = tl.tucker_to_tensor((core, factors))

    # Plotting all matrices and tensors on a common scale with a single color bar
    def plot_with_common_colorbar(tensors, titles):
        num_plots = len(tensors)
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
        
        # Find global min and max for common color scale
        vmin = min(tensor.min() for tensor in tensors)
        vmax = max(tensor.max() for tensor in tensors)
        
        for ax, tensor, title in zip(axes, tensors, titles):
            im = ax.imshow(tensor, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(title)
        
        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.tight_layout()
        plt.show()

    # Plot factor matrices
    plot_with_common_colorbar([factors[0], factors[1], factors[2]],
                              ["Temporal Factor Matrix (A)", "Row Factor Matrix (B)", "Column Factor Matrix (C)"])

    # Plot core tensor slices
    plot_with_common_colorbar([core[i] for i in range(core.shape[0])],
                              [f"Core Tensor Slice {i+1}" for i in range(core.shape[0])])

    # Plot original and reconstructed tensor slices
    plot_with_common_colorbar([V, V_reconstructed], ["Original Tensor Slices", "Reconstructed Tensor Slices"])

if __name__ == "__main__":
    main()