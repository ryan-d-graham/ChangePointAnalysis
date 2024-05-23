import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks
import os
import threading
import keyboard
import winsound  # For beep sound on Windows

# Initialize global variables
timestamps = []
weights = []
change_points = []
weighted_rates = []
update_plot = False

def bayesian_blocks_wrapper(data, p0=0.05, weights=None):
    data = np.asarray(data)
    if weights is not None:
        weights = np.asarray(weights)
    edges = bayesian_blocks(data, p0=p0, fitness='events', weights=weights)
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
    global timestamps, weights, update_plot
    start_time = time.time()
    
    while True:
        if keyboard.is_pressed('enter'):
            current_time = time.time()
            timestamps.append(current_time - start_time)
            weight = input("Enter weight for the meal (or press Enter for default): ")
            weights.append(float(weight) if weight else 1.0)
            update_analysis()
            while keyboard.is_pressed('enter'):
                pass  # Avoid multiple recordings
        
        if keyboard.is_pressed('p'):
            update_plot = True

def update_analysis():
    global change_points, weighted_rates
    if len(timestamps) > 0:
        # Ensure the lengths of timestamps and weights are the same before proceeding
        if len(timestamps) != len(weights):
            print("Error: The lengths of timestamps and weights are not the same.")
            return
        
        new_change_points = bayesian_blocks_wrapper(timestamps, weights=weights)
        new_weighted_rates = calculate_poisson_rates(timestamps, weights, new_change_points)
        
        if len(new_change_points) > len(change_points):
            change_points[:] = new_change_points
            weighted_rates[:] = new_weighted_rates
            alert_change_point()
        
        change_points[:] = new_change_points
        weighted_rates[:] = new_weighted_rates

        display_analysis()

def alert_change_point():
    print("Change point detected!")
    # Beep sound (Windows only)
    winsound.Beep(1000, 500)

def display_plot():
    plt.clf()
    plt.plot(timestamps, np.ones_like(timestamps), 'b.', markersize=10, label='Key Presses')
    for i, edge in enumerate(change_points):
        plt.axvline(edge, color='r', linestyle='--', label='Change Point' if i == 0 else "")
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        plt.hlines(weighted_rates[i], start, end, colors='g', linestyles='-', label='Poisson Rate' if i == 0 else "")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Activity / Poisson Rate')
    plt.title('Key Press Activity with Bayesian Blocks Change Points and Poisson Rates')
    plt.legend()
    plt.draw()
    plt.pause(0.001)  # Update the plot interactively

def display_analysis():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{'Block Start':<15}{'Block End':<15}{'Weighted Rate':<15}{'Delta W':<15}")
    print("="*60)
    delta_ws = calculate_delta_w(weighted_rates)
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        rate = weighted_rates[i]
        delta_w = delta_ws[i - 1] if i > 0 else 0
        print(f"{start:<15.2f}{end:<15.2f}{rate:<15.2f}{delta_w:<15.2f}")
    
    if len(timestamps) > 0:
        current_time = time.time() - timestamps[0]
        cumulative_weight = calculate_cumulative_weight(timestamps, weights, change_points, current_time)
        current_rate = np.sum(weights) / current_time
        print("\nCurrent Weighted Rate (approaching 0 asymptotically):", current_rate)
        print("Cumulative Weight up to current time:", cumulative_weight)

def main():
    global update_plot  # Declare update_plot as global to modify it within the function
    
    # Start the key press recording thread
    key_press_thread = threading.Thread(target=record_key_presses)
    key_press_thread.start()
    
    plt.ion()  # Turn on interactive mode
    plt.show()
    
    while True:
        time.sleep(1)  # Refresh the analysis every second
        if timestamps:
            update_analysis()
        if update_plot:
            display_plot()
            update_plot = False

if __name__ == "__main__":
    main()