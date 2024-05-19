from astropy.stats import bayesian_blocks as bb
from numpy import histogram as hist
from numpy import diff
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from os import system as sys

ttedata = [0.0,]
store_open = datetime(2024, 5, 18, 10, 30, 0, 0)
begin = datetime.now()
event = ""
while event != "q":
	sys('cls')
	print("Change-Point and Event Rate Tracker\n")
	print("Print a readout with bar chart (p)\n")
	print("Add time to event tracker (t)\n")
	print("Remove the last time (r)\n")
	for n, e in enumerate(ttedata):
		print("Event ", str(n), ": ", str(round(e/60.0, 2)))
	event = input(">>")
	if event == "t":
		ttedata.append((datetime.now()-begin).total_seconds())
		#ttedata.append((datetime.now()-store_open).total_seconds())
	if event == "p":
		ttedata.append((datetime.now()-begin).total_seconds())
		#ttedata.append((datetime.now()-store_open).total_seconds())
		change_points = bb(ttedata, fitness='events', ncp_prior = 0.5)
		block_sizes = diff(change_points)
		counts, _ = hist(ttedata, bins = change_points)
		rates = counts / block_sizes
		current_rate = round(60*rates[-1], 2)
		for n, cp in enumerate(change_points):
			print("Change-point ", str(n), ": ", str(round(cp/60.0, 2)), "\n")
		print("Current Event Rate: ", current_rate, " events / min.\n")
		plt.bar(x = change_points[0:-1]/60.0, height = 60*rates, width = block_sizes/60.0, align = 'edge', alpha = 0.5)
		plt.vlines(change_points[1:-1]/60.0, 0.0, 60*max(rates), color = 'r')
		plt.title("Event Rate History")
		plt.xlabel("Time (minutes)")
		plt.ylabel("Events/Min")
		plt.show()
		ttedata.pop() #this was not a real event, but a query on current rate if an event happened right now
	if event == "r":
		ttedata.pop()   # use this if an event was marked by accident
