import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from astropy.stats import bayesian_blocks
import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bayesian Blocks and NMF Analysis')
    parser.add_argument('--data_model', type=str, choices=['events', 'regular_events', 'measures'], default='measures', 
                        help='Bayesian Blocks data model type: events, regular_events, or measures')
    parser.add_argument('--nmf_components', type=int, default=3, help='Number of NMF components')
    parser.add_argument('--nmf_max_iter', type=int, default=200, help='Maximum number of NMF iterations')
    return parser.parse_args()

def load_seizure_data():
    # Replace this with the actual path or method to load your seizure identification dataset
    data = pd.read_csv('path/to/your/seizure_dataset.csv')
    time_tags = data['time'].values
    measurements = data.drop(columns=['time']).values
    return time_tags, measurements

def main():
    args = parse_arguments()
    
    # Load the seizure identification dataset
    time_tags, measurements = load_seizure_data()

    # Apply Bayesian Blocks to find change points
    edges = bayesian_blocks(t=time_tags, x=measurements, fitness=args.data_model)
    print("Detected change points:", edges)

    # Create a matrix to store the weighted rates for each block
    num_variables = measurements.shape[1]
    num_blocks = len(edges) - 1
    V = np.zeros((num_variables, num_blocks))

    # Calculate the weighted rate for each variable in each block
    for i in range(num_variables):
        for j in range(num_blocks):
            start, end = edges[j], edges[j + 1]
            mask = (time_tags >= start) & (time_tags < end)
            duration = end - start
            if np.sum(mask) > 0 and duration > 0:
                V[i, j] = np.sum(measurements[mask, i]) / duration
            else:
                V[i, j] = 0  # Handle zero counts or zero duration safely

    # Apply Non-Negative Matrix Factorization (NMF)
    model = NMF(n_components=args.nmf_components, max_iter=args.nmf_max_iter, init='random', random_state=0)
    W = model.fit_transform(V)
    H = model.components_
    V_star = np.dot(W, H)

    # Visualization using seaborn heatmaps with borders and integer axis ticks
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original matrix V
    sns.heatmap(V, ax=axes[0, 0], cmap='inferno', linewidths=.5, linecolor='white', cbar=True, 
                xticklabels=np.arange(1, V.shape[1] + 1), yticklabels=np.arange(1, V.shape[0] + 1))
    axes[0, 0].set_title('Input to NMF: Original Matrix V')
    axes[0, 0].set_ylabel('Variables')
    axes[0, 0].set_xlabel('Blocks')

    # Basis matrix W
    sns.heatmap(W, ax=axes[0, 1], cmap='inferno', linewidths=.5, linecolor='white', cbar=True, 
                xticklabels=np.arange(1, W.shape[1] + 1), yticklabels=np.arange(1, W.shape[0] + 1))
    axes[0, 1].set_title('Output of NMF: Basis Matrix W')
    axes[0, 1].set_ylabel('Variables')
    axes[0, 1].set_xlabel('Latent Features')

    # Coefficient matrix H
    sns.heatmap(H, ax=axes[1, 0], cmap='inferno', linewidths=.5, linecolor='white', cbar=True, 
                xticklabels=np.arange(1, H.shape[1] + 1), yticklabels=np.arange(1, H.shape[0] + 1))
    axes[1, 0].set_title('Output of NMF: Coefficient Matrix H')
    axes[1, 0].set_ylabel('Latent Features')
    axes[1, 0].set_xlabel('Blocks')

    # Reconstructed matrix V*
    sns.heatmap(V_star, ax=axes[1, 1], cmap='inferno', linewidths=.5, linecolor='white', cbar=True, 
                xticklabels=np.arange(1, V_star.shape[1] + 1), yticklabels=np.arange(1, V_star.shape[0] + 1))
    axes[1, 1].set_title('Reconstructed Matrix V*')
    axes[1, 1].set_ylabel('Variables')
    axes[1, 1].set_xlabel('Blocks')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
