import time
import numpy as np
import os
import threading
import keyboard
from astropy.stats import bayesian_blocks

# Initialize global variables
timestamps = []
weights = []

def bayesian_blocks_wrapper(data, p0=0.05, weights=None):
    data = np.asarray(data)
    if weights is not None:
        weights = np.asarray(weights)
    edges = bayesian_blocks(t=data, x=weights, p0=p0, fitness='events')
    return edges

def calculate_poisson_rates(timestamps, weights, edges):
    rates = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        duration = end - start
        total_weight = np.sum(weights[(timestamps >= start) & (timestamps < end)])
        rate = total_weight / duration if duration > 0 else 0
        rates.append(rate)
    return rates

def calculate_delta_w(rates):
    delta_w = np.diff(rates)
    return delta_w

def calculate_cumulative_weight(timestamps, weights, edges, time_t):
    cumulative_weight = 0
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if time_t < start:
            break
        if time_t <= end:
            duration = time_t - start
            rate = np.sum(weights[(timestamps >= start) & (timestamps < end)]) / (end - start)
            cumulative_weight += rate * duration
            break
        else:
            duration = end - start
            rate = np.sum(weights[(timestamps >= start) & (timestamps < end)]) / duration
            cumulative_weight += rate * duration
    return cumulative_weight

def record_key_presses():
    global timestamps, weights
    start_time = time.time()
    
    while True:
        if keyboard.is_pressed('enter'):
            current_time = time.time()
            timestamps.append(current_time - start_time)
            weight = input("Enter weight for the meal (or press Enter for default): ")
            weights.append(float(weight) if weight else 1.0)
            display_statistics()
            while keyboard.is_pressed('enter'):
                pass  # Avoid multiple recordings

def update_analysis():
    global change_points, weighted_rates
    if len(timestamps) > 0:
        # Ensure the lengths of timestamps and weights are the same before proceeding
        if len(timestamps) != len(weights):
            print("Error: The lengths of timestamps and weights are not the same.")
            return
        
        new_change_points = bayesian_blocks_wrapper(timestamps, weights=weights)
        new_weighted_rates = calculate_poisson_rates(timestamps, weights, new_change_points)
        
        change_points[:] = new_change_points
        weighted_rates[:] = new_weighted_rates

def display_statistics():
    global change_points, weighted_rates
    update_analysis()
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Business Activity Tracker\n")
    print("Instructions:")
    print("- Press Enter to record a unit of business activity.")
    print("- Optionally, enter a weight for the activity.")
    print("- The program calculates statistics based on recorded activity.\n")
    
    if len(timestamps) > 0:
        delta_ws = calculate_delta_w(weighted_rates)
        print(f"{'Block Start':<15}{'Block End':<15}{'Weighted Rate':<15}{'Delta W':<15}")
        print("="*60)
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            rate = weighted_rates[i]
            delta_w = delta_ws[i - 1] if i > 0 else 0
            print(f"{start:<15.2f}{end:<15.2f}{rate:<15.2f}{delta_w:<15.2f}")
        
        current_time = time.time() - timestamps[0]
        cumulative_weight = calculate_cumulative_weight(timestamps, weights, change_points, current_time)
        current_rate = np.sum(weights) / current_time
        print("\nCurrent Weighted Rate (approaching 0 asymptotically):", current_rate)
        print("Cumulative Weight up to current time:", cumulative_weight)
    else:
        print("No data recorded yet.")

def main():
    global update_plot  # Declare update_plot as global to modify it within the function
    
    # Start the key press recording thread
    key_press_thread = threading.Thread(target=record_key_presses)
    key_press_thread.start()
    
    while True:
        time.sleep(1)  # Refresh the display every second
        display_statistics()

if __name__ == "__main__":
    main()
