import keyboard
import time
import numpy as np
from astropy.stats import bayesian_blocks as bb
import matplotlib.pyplot as plt

def bayesian_blocks_wrapper(data, p0=0.05, weights=None):
  data = np.asarray(data)
  if weights is not None:
    weights = np.asarray(weights)
    edges = bayesian_blocks(data, p0, fitness='events', weights=weights)
    return edges

def record_key_presses(duration=60):
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
    count = np.sum((timestamps >= start) & (timestamps < end)])
    rate = total_weight / duration if duration > 0 else 0
    rates.append(rate)
  return rates

def main():
  #Record key presses for a specified duration (e.g., 60 seconds)
  duration = 60 # in seconds
  timestamps, weights = record_key_presses(duration)

  if len(timestamps) > 0:
    edges = bayesian_blocks_wrapper(timestamps)
    rates = calculate_poisson_rates(timestamps, weights, edges)
    print("Bayesian Blocks edges:", edges)
    print("Poisson rates per segment:", rates)

    #plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, np.ones_like(timestamps), 'b.', markersize=10, label='Key Presses')
    for i, edges in enumerate(edges):
      plt.axvline(edge, color='r', linestyle='--', label='Change Point' if i == 0 else "")
      plt.xlabel('Time (seconds)')
      plt.ylabel('Activity / Poisson Rate')
      plt.title('Key Press Activity with Bayesian Blocks Change Points and Poisson Rates')
      plt.legend()
      plt.show()
  else:
    print("No key presses were recorded.")

if __name__ == "__main__":
  main()


