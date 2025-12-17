# plot_rewards.py
import numpy as np
import matplotlib.pyplot as plt

# Rewards laden (von train_qlearning.py erzeugt)
rewards = np.load("rewards.npy")

plt.figure(figsize=(10,5))
plt.plot(rewards, label="Reward pro Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Lernkurve – Princess RL Agent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()
