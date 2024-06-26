import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    parser.add_argument('--num_observations', type=int, default=10, help='Number of observations in each channel')
    return parser.parse_args()

def load_example_data(epsilon, mea_channels, num_observations):
    # Generate synthetic MEA data with inhomogeneous timestamps
    np.random.seed(42)
    timestamps = [np.sort(np.random.uniform(epsilon, 10, num_observations)) for _ in range(mea_channels)]

    # Generate lambda values from a gamma distribution
    shape, scale = 2.0, 1.0
    lambdas = np.random.gamma(shape, scale, (num_observations, mea_channels))

    # Generate Poisson-distributed data using the lambda matrix and ensure positive integers
    measurements = np.random.poisson(lam=lambdas) + 1  # Shift by 1 to ensure all values are positive integers

    return timestamps, measurements

def main():
    args = parse_arguments()
    ranks = tuple(map(int, args.nntd_rank.split(',')))

    # Load example data
    mea_channels = args.mea_rows * args.mea_cols
    timestamps, measurements = load_example_data(args.epsilon, mea_channels, args.num_observations)

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

    # Check for non-zero matrices
    print("Non-zero elements in original tensor:", np.count_nonzero(V))
    print("Non-zero elements in reconstructed tensor:", np.count_nonzero(V_reconstructed))

    # Plot temporal, row, and column factors independently with annotated axes and ticks starting from 1
    def plot_factor(factor, title, xlabel, ylabel, xticks, yticks):
        plt.figure(figsize=(10, 8))
        sns.heatmap(factor, cmap='viridis', linewidths=.5, linecolor='white',
                    xticklabels=xticks, yticklabels=yticks)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(np.arange(len(xticks)), xticks)
        plt.yticks(np.arange(len(yticks)), yticks)
        plt.show()

    temporal_ticks = np.arange(1, factors[0].shape[1] + 1)
    row_ticks = np.arange(1, factors[1].shape[0] + 1)
    column_ticks = np.arange(1, factors[2].shape[0] + 1)

    plot_factor(factors[0], "Temporal Factor Matrix (A)", "Temporal Components", "Temporal Blocks", temporal_ticks, np.arange(1, factors[0].shape[0] + 1))
    plot_factor(factors[1], "Row Factor Matrix (B)", "Latent Row Components", "Row Channel", np.arange(1, factors[1].shape[1] + 1), row_ticks)
    plot_factor(factors[2], "Column Factor Matrix (C)", "Latent Column Components", "Column Channel", np.arange(1, factors[2].shape[1] + 1), column_ticks)

    # Plot core tensor slices in a rectangular grid on a common scale with a color bar placed outside the grid
    def plot_core_slices(core, title):
        num_slices = core.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_slices)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
        
        # Find global min and max for common color scale
        vmin = core.min()
        vmax = core.max()
        
        for i in range(grid_size * grid_size):
            ax = axes.flat[i]
            if i < num_slices:
                im = ax.imshow(core[i, :, :], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_title(f'Slice {i+1}')
                ax.set_xticks(np.arange(core.shape[2]) + 1)
                ax.set_yticks(np.arange(core.shape[1]) + 1)
            else:
                ax.axis('off')
        
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', pad=0.02)
        plt.suptitle(title)
        plt.show()

    plot_core_slices(core, "Core Tensor Slices")

if __name__ == "__main__":
    main()