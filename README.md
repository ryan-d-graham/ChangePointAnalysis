ChangePointAnalysis

Overview

ChangePointAnalysis is a Python tool for real-time change-point detection and rate monitoring in time-tagged event data using the Bayesian Blocks algorithm (Scargle, 2013).

Features

	•	Real-time Data Collection: Record timestamps and associated weights via keyboard inputs.
	•	Bayesian Blocks Analysis: Detect change points in event rates.
	•	Poisson Rate Calculation: Compute the event rate for segments between change points.
	•	Dynamic Visualization: Generate plots displaying key presses, change points, and event rates.

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

Example Output

The script will display Bayesian Blocks edges, Poisson rates per segment, and plot the results, including key presses, change points, and event rates.

Contributing

We welcome contributions! Please fork the repository and create a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements

This tool utilizes the Bayesian Blocks implementation from the Astropy library.

Feel free to adjust the README to better fit your project’s specific details and additional features.