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
11. **Neuroscience**: Analyzing EEG data to detect and analyze changes in brain activity.

#### Patient Monitoring

**Goal**: Early detection of changes in patient health status.

**Implementation**:
- **Data Collection**: Continuously monitor vital signs such as heart rate, blood pressure, and oxygen levels.
- **Change Point Detection**: Use Bayesian Blocks to identify significant shifts in these metrics.
- **Pattern Recognition**: Apply NMF to decompose the data into latent health patterns, distinguishing between normal and abnormal states.

**Benefits**:
- **Real-Time Alerts**: (Trigger) Prompt healthcare providers to potential health issues.
- **Personalized Care**: (Retrospective) Tailor interventions based on individual patient data patterns.

#### Epidemiology

**Goal**: Understand and control disease spread.

**Implementation**:
- **Data Collection**: Gather data on infection rates, recoveries, and fatalities across regions and time.
- **Change Point Detection**: Use Bayesian Blocks to detect significant changes in disease trends.
- **Pattern Recognition**: Apply NMF to identify latent factors contributing to the spread, such as demographic or environmental factors.

**Benefits**:
- **Timely Interventions**: Inform public health responses to emerging outbreaks.
- **Policy Making**: Guide resource allocation and preventive measures based on identified patterns.

#### Analyzing EEG Data

**Goal**: Detect and analyze changes in brain activity over time.

**Implementation**:
- **Data Collection**: Record EEG signals from multiple electrodes over time.
- **Change Point Detection**: Use Bayesian Blocks to identify significant changes in EEG signal patterns.
- **Matrix Construction**: Construct a matrix where each row corresponds to an EEG channel and each column corresponds to a time block defined by detected change points.
- **Pattern Recognition**: Apply NMF to decompose the EEG data matrix into basis and coefficient matrices, uncovering latent patterns and components in the EEG signals.

**Benefits**:
- **Event Detection**: Identify and localize epileptic seizures, sleep stages, or cognitive events.
- **Pattern Recognition**: Discover underlying neural patterns and rhythms.
- **Feature Extraction**: Extract meaningful features for further analysis or machine learning applications.

### Bayesian Blocks: Parameter p0

**p0**: The false positive rate, which determines the sensitivity of the algorithm to detecting change points.

- **Increasing p0**:
  - **Consequence**: Detects fewer change points.
  - **Use Case**: Suitable for applications where only significant changes are of interest, reducing the risk of false positives.

- **Decreasing p0**:
  - **Consequence**: Detects more change points.
  - **Use Case**: Suitable for applications requiring high sensitivity to changes, even if it increases the risk of false positives.

- **Tuning p0**:
  - Adjust **p0** based on the specific requirements of the application to balance sensitivity and false positive rate.
  - Experiment with different values of **p0** to find the optimal setting for your dataset.

## Usage

### Running BayesRate.py

To detect change points using the Bayesian method, configure the script parameters within the script itself, then run it using Python.

### Running GoldSpeed.py

To detect changes in event rates using GoldSpeed, configure the script parameters within the script itself, then run it using Python.

### Running BayesBlocksNMF.py

To detect common change points and analyze multivariate event data using NMF, configure the script parameters within the script itself, then run it using Python.

### Example Output

The scripts output the detected change points to the terminal and visualize results using Matplotlib.