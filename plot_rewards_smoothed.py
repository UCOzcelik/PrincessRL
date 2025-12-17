# plot_rewards_smoothed.py
import numpy as np
import matplotlib.pyplot as plt

# Rewards laden
rewards = np.load("rewards.npy")

# Moving Average Funktion
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Gleitender Durchschnitt (100 Episoden Fenster)
smoothed = moving_average(rewards, window_size=100)

plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.3, label="Reward pro Episode (raw)")
plt.plot(smoothed, color="red", linewidth=2, label="Geglättete Lernkurve (Moving Average 100)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Geglättete Lernkurve – Princess RL Agent")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve_smoothed.png")
plt.show()
