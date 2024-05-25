import keyboard
import time
import numpy as np
from astropy.stats import bayesian_blocks as bb
import matplotlib.pyplot as plt

def bayesian_blocks_wrapper(data, p0=0.05, weights=None):
    data = np.asarray(data)
    if weights is not None:
        weights = np.asarray(weights)
    edges = bb(t=data, x=weights, p0=p0, fitness='events')
    return edges

def record_key_presses(duration=36000):
    timestamps = []
    weights = []
    start_time = time.time()
    end_time = start_time + duration
    print("Press Enter key to record a timestamp. Enter weight if needed. Press Esc key to stop early.")
    while time.time() < end_time:
        if keyboard.is_pressed('enter'):
            current_time = time.time()
            timestamps.append(current_time - start_time)
            weight = input("Enter weight for the meal (or press Enter for default): ")
            weights.append(float(weight) if weight else 1.0)
            print(f"Recorded timestamp: {current_time - start_time:.2f} seconds, Weight: {weights[-1]}")
            while keyboard.is_pressed('enter'):
                pass
        if keyboard.is_pressed('esc'):
            break
    return np.array(timestamps), np.array(weights)

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
    if len(rates) > 1:
        delta_ws = np.diff(rates)
    else:
        delta_ws = np.array([])
    return delta_ws

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

def main():
    # Record key presses for a specified duration (e.g., 36000 seconds or 10 hours)
    duration = 36000 # in seconds
    timestamps, weights = record_key_presses(duration)
    p0 = float(input("Set sensitivity p0: "))
    if len(timestamps) > 0:
        edges = bayesian_blocks_wrapper(timestamps, p0, weights)
​⬤