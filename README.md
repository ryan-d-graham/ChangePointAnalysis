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
3. **Matrix Construction**: Constructs a matrix of weighted rates for each block and variable.
4. **Non-Negative Matrix Factorization (NMF)**: Decomposes the matrix into basis and coefficient matrices using NMF.
5. **Visualization**: Visualizes the original matrix, the basis matrix, the coefficient matrix, and the reconstructed matrix using heatmaps.

### Applications

This script can be used for:

1. **Event Analysis**: Identifying change points in time-series data.
2. **Pattern Recognition**: Decomposing and analyzing multivariate data to uncover latent structures.
3. **Data Segmentation**: Segmenting data into meaningful blocks based on event rates and visualizing the results.
4. **Genomics**: Gene expression analysis, copy number variation detection.
5. **Finance**: Market analysis, risk management.
6. **Environmental Science**: Climate change detection, ecosystem monitoring.
7. **Healthcare**: Patient monitoring, epidemiology.
8. **Marketing and Sales**: Consumer behavior analysis, sales data analysis.
9. **Manufacturing**: Process monitoring, equipment maintenance.
10. **Social Sciences**: Sociological studies, political science.

## Usage

### Running BayesRate.py

To detect change points using the Bayesian method, configure the script parameters within the script itself, then run it using Python.

### Running GoldSpeed.py

To detect changes in event rates using GoldSpeed, configure the script parameters within the script itself, then run it using Python.

### Running BayesBlocksNMF.py

To detect common change points and analyze multivariate event data using NMF, configure the script parameters within the script itself, then run it using Python.

### Example Output

The scripts output the detected change points to the terminal and visualize results using Matplotlib.