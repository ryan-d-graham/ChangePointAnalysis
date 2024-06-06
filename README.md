# ChangePointAnalysis

## BB_SNNTD.py

This script performs the following tasks:
1. **Data Preparation**: Processes multivariate MEA data, combining timestamps and weights from different channels.
2. **Change Point Detection**: Uses the Bayesian Blocks algorithm to detect common change points across all channels.
3. **Tensor Construction**: Constructs a tensor of weighted rates for each block and channel.
4. **Tensor Decomposition**: Decomposes the tensor into core and factor matrices using Sparse Non-Negative Tucker Decomposition (SNNTD).
5. **Visualization**: Visualizes the core tensor slices and the factor matrices using heatmaps.

### Command-Line Arguments

- `--nntd_rank`: Comma-separated ranks for the SNNTD.
- `--p0`: False alarm probability for Bayesian Blocks.
- `--epsilon`: Epsilon value to avoid zero timestamps.
- `--mea_rows`: Number of rows in the MEA grid.
- `--mea_cols`: Number of columns in the MEA grid.
- `--sparsity`: Sparsity threshold for enforcing sparsity in decomposition.
- `--num_observations`: Number of observations in each channel.

### Example Usage

```sh
python BB_SNNTD.py --nntd_rank 10,5,5 --p0 0.05 --epsilon 1e-10 --mea_rows 8 --mea_cols 8 --sparsity 0.1 --num_observations 10

Applications

Micro-electrode Arrays (MEAs)

Goal: Detect and analyze changes in neural activity over time using MEAs.

Implementation:

	•	Data Collection: Record electrical activity from multiple electrodes over time using MEAs.
	•	Change Point Detection: Use Bayesian Blocks to identify significant changes in neural signal patterns.
	•	Tensor Construction: Construct a tensor where each dimension corresponds to a time block defined by Bayesian Blocks and each entry represents the weighted rate of neural activity for each MEA channel within these blocks.
	•	Step-by-step:
	•	Segmenting Data: Use Bayesian Blocks to segment the timestamps of neural activity.
	•	Calculating Weighted Rates: For each segment and MEA channel, calculate the weighted rate of neural activity, constructing a 3D tensor where dimensions represent time blocks, MEA rows, and MEA columns.
	•	Pattern Recognition: Apply SNNTD to decompose the data tensor into core and factor matrices, uncovering latent neural patterns and components.

Benefits:

	•	Event Detection: Identify and localize neural firing patterns and events.
	•	Pattern Recognition: Discover underlying neural circuits and rhythms.
	•	Feature Extraction: Extract meaningful features for further analysis in neurological research.

Key Matrices

	•	Temporal Factor Matrix: Represents temporal blocks and temporal components, revealing how different time segments contribute to the neural activity patterns.
	•	Row Factor Matrix: Represents row channels and latent row components, indicating how different MEA rows contribute to the observed activity.
	•	Column Factor Matrix: Represents column channels and latent column components, showing how different MEA columns contribute to the observed activity.
	•	Core Tensor Slices: Represent interactions between temporal, row, and column components, providing a detailed view of the neural activity’s structure.

Bayesian Blocks: Parameter p0

	•	p0: The false positive rate, determining the sensitivity of the algorithm to detecting change points.
	•	Increasing p0: Detects fewer change points; suitable for applications where only significant changes are of interest, reducing the risk of false positives.
	•	Decreasing p0: Detects more change points; suitable for applications requiring high sensitivity to changes, even if it increases the risk of false positives.
	•	Tuning p0: Adjust based on the specific requirements of the application to balance sensitivity and false positive rate. Experiment with different values to find the optimal setting for your dataset.

Usage

Running BB_SNNTD.py

To detect common change points and analyze multivariate MEA data using SNNTD, run the script using the command line arguments above.