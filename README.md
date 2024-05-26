ChangePointAnalysis

Overview

ChangePointAnalysis is a Python tool for real-time change-point detection and rate monitoring in time-tagged event data using the Bayesian Blocks algorithm (Scargle, 2013).

Features

	•	Real-time Data Collection: Record timestamps and associated weights via keyboard inputs.
	•	Bayesian Blocks Analysis: Detect change points in weighted event rates.
	•	Poisson Rate Calculation: Compute the weighted event rate for segments between change points.
	•	Dynamic Visualization: Generate plots displaying key presses, change points, and weighted event rates.
 
The weights may correspond to a feature or observable of interest, such as quantity or weight of chicken, sales, or perhaps something more abstract...
I find it an interesting prospect to analyze a business' raw unweighted event rates and then plot this against the same events using various distinct
weighting schemas to compare change-points in each weight/feature space...

Installation

Clone the repository:

git clone https://github.com/ryan-d-graham/ChangePointAnalysis.git
cd ChangePointAnalysis

Install the required dependencies:

pip install numpy astropy matplotlib keyboard

Usage

Run the main script to start recording key presses and detecting change points:

python ChangePointAnalysis.py

Commands

	•	Press Enter: Record a timestamp and input a weight for the event.
	•	Press Esc: Stop recording key presses.
	•	Set sensitivity p0: Choose the sensitivity of change-point detection.

Nota Bene on p0:

In the context of the Bayesian Blocks algorithm,  p0  represents the false-positive rate, or the probability of detecting a change point when there is none. This parameter controls the sensitivity of the algorithm to changes in the data. A lower  p0  value means fewer false positives, leading to fewer detected change points and a smoother model. Conversely, a higher  p0  value allows for more detected change points, potentially capturing more nuances in the data but also increasing the risk of false positives.

Choosing an appropriate  p0  depends on the specific application and the acceptable trade-off between sensitivity and specificity.

Example Output

The script will display Bayesian Blocks edges, Poisson rates per segment, and plot the results, including key presses, change points, and event rates.

Contributing

We welcome contributions! Please fork the repository and create a pull request.

License

This project is licensed under the MIT License (LOL, just kidding). See the LICENSE file for details :P

Acknowledgements

This tool utilizes the Bayesian Blocks implementation from the Astropy library. 

Further reading

I read this article more than once and then some in order to fully grasp and gain ownership of this algorithm. You don't have to do this, but I strongly recommend it. You will be better off having done so. 
https://iopscience.iop.org/article/10.1088/0004-637X/764/2/167/pdf

