# ChangePointAnalysis

## Projects

### BayesRate.py

This script utilizes Bayesian methods to detect change points in time series data. By applying probabilistic models, it identifies the most probable locations of structural breaks. Note: Currently, BayesRate.py does not take command line arguments and needs to be configured within the script itself.

### GoldSpeed.py

Implements a method to detect changes in the rate of events, providing a detailed analysis of temporal event sequences. This script also needs to be configured within the script itself for input and output paths.

### BayesBlocksNMF.py

This script performs the following tasks:

1. **Data Preparation**: Processes multivariate event data, combining timestamps and weights from different variables.
2. **Change Point Detection**: Uses the Bayesian Blocks algorithm to detect common change points across all variables.
3. **Matrix Construction**: Constructs a matrix of weighted poisson rates for each block and variable.
4. **Non-Negative Matrix Factorization (NMF)**: Decomposes the matrix into basis and coefficient matrices using NMF.
5. **Visualization**: Visualizes the original matrix, the basis matrix, the coefficient matrix, and the reconstructed matrix using heatmaps.

This script can be used for event analysis, pattern recognition, and data segmentation by identifying change points in time-series data, uncovering latent structures, and segmenting data into meaningful blocks.

## Usage

### Running BayesRate.py

To detect change points using the Bayesian method, configure the script parameters within the script itself, then run it using Python.

### Running GoldSpeed.py

To detect changes in event rates using GoldSpeed, configure the script parameters within the script itself, then run it using Python.

### Running BayesBlocksNMF.py

To detect common change points and analyze multivariate event data using NMF, configure the script parameters within the script itself, then run it using Python.

### Example Output

The scripts output the detected change points to the terminal and visualize results using Matplotlib.