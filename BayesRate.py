from astropy.stats import bayesian_blocks as bb
from numpy import histogram as hist
from numpy import diff
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from os import system as sys
import pickle

ttedata = []
#store_open = datetime(2024, 5, 18, 10, 30, 0, 0)

try:
	with open("ttedata.pkl", 'rb') as file:
		ttedata = pickle.load(file)
except:
	print("Could not load TTE file...\n")  
	
try:
	with open("resume.pkl", 'rb') as file:
		begin = pickle.load(file)
except:
	print("Could not load last TTE for resuming...\n")
	print("Using current system time as beginning...\n")
	begin = datetime.now()
	

event = ""
p0 = 0.05 # default
while event != "q":
	sys('cls')
	print("Change-Point and Event Rate Tracker\n")
	print("Plot Rate History (p)\n")
	print("Display Statistics (d)\n")
	print("Save times to file (s)\n")
	print("Load times from file (l)\n")
	print("View Event Times (t)\n")
	print("Reset Everything (R)\n")
	print("Remove last event (r)\n")
	print("Adjust p0 (n)\n")
	print("Quit Program (q)\n")
	event = input("Press Enter to log a TTE>>")
	if event == "R":
		ttedata = []
		begin = datetime.now()
		with open("ttedata.pkl", 'wb') as file:
			pickle.dump(ttedata, file)
		with open("resume.pkl", 'wb') as file:
				pickle.dump(datetime.now(), file)
	if event == "l":
		try:
			with open("ttedata.pkl", 'rb') as file:
				ttedata = pickle.load(file)
		except:
			print("Could not load TTE file...\n")
			input("Press Enter to continue...\n")
		try:
			with open("resume.pkl", 'wb') as file:
				pickle.dump(datetime.now(), file)
		except:
			print("Could not load save time...\n")
			input("Press Enter to continue...\n")
	if event == "s":
		with open("ttedata.pkl", 'wb') as file:
			pickle.dump(ttedata, file)
		with open("resume.pkl", 'wb') as file:
			pickle.dump(datetime.now(), file)
	if event == "n":
		sys('cls')
		print("p0 = ", str(p0), "\n")
		p0 = float(input("Enter new value: "))
	if event == "t":
		sys('cls')
		try:
			for n, e in enumerate(ttedata):
				print("Event ", str(n+1), ": ", str(round(e/60.0, 3)))
			input("Press Enter to continue...\n")
		except:
			print("Not enough data...\n") 
	if event == "d":
		sys('cls')
		ttedata.append((datetime.now()-begin).total_seconds())
		#ttedata.append((datetime.now()-store_open).total_seconds())
		try:
			change_points = bb(ttedata, fitness='events', p0 = p0)
			block_sizes = diff(change_points)
			counts, _ = hist(ttedata, bins = change_points)
			rates = counts / block_sizes
			current_rate = round(60*rates[-1], 2)
			print("Change-Points: \n")
			for n, cp in enumerate(change_points, 1):
				print("CP", str(n+1), ": ", str(round(cp/60.0, 3)), " minutes")
			print("\nCurrent Rate: \n")
			print(str(current_rate), "events/min\n")
			ttedata.pop()
			input("Press Enter to continue...\n")
		except:
			print("\nBayesian Blocks failed... (not enough data?)\n")
			input("Press Enter to continue...\n")
	if event == "":
		ttedata.append((datetime.now()-begin).total_seconds())
		#ttedata.append((datetime.now()-store_open).total_seconds())
		with open('ttedata.pkl', 'wb') as file:
			pickle.dump(ttedata, file)
	if event == "p":
		try:
			ttedata.append((datetime.now()-begin).total_seconds())
			#ttedata.append((datetime.now()-store_open).total_seconds())
			change_points = bb(ttedata, fitness='events', p0 = p0)
			block_sizes = diff(change_points)
			counts, _ = hist(ttedata, bins = change_points)
			rates = counts / block_sizes
			current_rate = round(60*rates[-1], 2)
			for n, cp in enumerate(change_points, 1):
				print("CP", str(n+1), ": ", str(round(cp/60.0, 3)), " minutes")
			print("\nCurrent Rate: ", str(current_rate), " events / min.\n")
			plt.bar(x = change_points[0:-1]/60.0, height = 60*rates, width = block_sizes/60.0, align = 'edge', alpha = 0.5)
			plt.vlines(change_points/60.0, 0.0, 60*max(rates), color = 'r', alpha = 0.5)
			plt.title("Event Rate History")
			plt.xlabel("Time (minutes)")
			plt.ylabel("Events/Min")
			plt.show()
			ttedata.pop()
			#this was not a real event, but a query on current rate if an event happened right now
		except:
			print("\nBayesian Blocks failed... (not enough data?)\n")
			input("Press Enter to continue...\n")
	if event == "r":
		if len(ttedata) < 1:
			break
		else:
			ttedata.pop()
			with open('ttedata.pkl', 'wb') as file:
				pickle.dump(ttedata, file)
