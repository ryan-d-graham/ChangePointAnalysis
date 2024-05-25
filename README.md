ChangePointAnalysis

Overview

ChangePointAnalysis is a Python tool for real-time change-point detection and rate monitoring in time-tagged event data using the Bayesian Blocks algorithm (Scargle, 2013). This tool is particularly useful for applications such as tracking customer queue intensity, monitoring abrupt changes in radiation intensity, detecting churn rates, and more.

Features

	•	Real-time change-point detection: Identify abrupt changes in event rates as they happen.
	•	Weighted Poisson rates: Estimate the rate of events in each detected time segment.
	•	Interactive plotting: Visualize key press data, change points, and event rates dynamically.

Use Cases

	1.	Astronomy: Detecting changes in radiation intensity with particle detections logged as TTE data.
	2.	Customer Service: Monitoring times of abrupt changes in customer order rates to predict workload.
	3.	Call Centers: Identifying optimal times for handling calls based on call rate changes.
	4.	Traffic Analysis: Detecting changes in traffic flow rates for better traffic management.

Installation

Clone the repository:

git clone https://github.com/ryan-d-graham/ChangePointAnalysis.git
cd ChangePointAnalysis

Install the required dependencies:

pip install -r requirements.txt

Usage

Run the main script to start recording key presses and detecting change points:

python ChangePointAnalysis.py

Commands

	•	Press Enter: Record a timestamp. You will be prompted to enter a weight for the event.
	•	Press Esc: Stop recording key presses.
	•	Set sensitivity p0: Choose the sensitivity of change-point detection.

Example Output

The script will display Bayesian Blocks edges, Poisson rates per segment, and plot the results.

Contributing

We welcome contributions! Please fork the repository and create a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements

This tool utilizes the Bayesian Blocks implementation from the Astropy library.

Feel free to customize further as per your needs!