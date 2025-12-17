# train_qlearning.py
import numpy as np
from princess_env import PrincessEnv

# Umgebung erstellen
env = PrincessEnv()

n_states = env.n_states
n_actions = env.n_actions

# Q-Tabelle initialisieren
Q = np.zeros((n_states, n_actions))

# Hyperparameter
alpha = 0.15          # Lernrate
gamma = 0.97          # Discount-Faktor

num_episodes = 15000  # mehr Episoden für schwierigeres, leicht zufälliges Environment

epsilon_start = 1.0
epsilon_end = 0.05

# lineares Epsilon-Decay
epsilon_decay = (epsilon_start - epsilon_end) / num_episodes

rewards_history = []
success_history = []

for ep in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0.0
    success = False

    # aktuelles Epsilon berechnen
    epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)

    while not done:
        # epsilon-greedy Policy
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # Q-Learning-Update
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state

        if info.get("success", False):
            success = True

    rewards_history.append(total_reward)
    success_history.append(1 if success else 0)

    # alle 500 Episoden etwas Statistik ausgeben
    if (ep + 1) % 500 == 0:
        last100_rewards = rewards_history[-100:]
        last100_success = success_history[-100:]
        avg_reward = np.mean(last100_rewards)
        avg_success = np.mean(last100_success) * 100.0
        print(
            f"Episode {ep+1}/{num_episodes} - "
            f"avg reward (last 100): {avg_reward:.2f}, "
            f"success rate (last 100): {avg_success:.1f}%, "
            f"epsilon={epsilon:.2f}"
        )

# Q-Tabelle speichern
np.save("q_table.npy", Q)
print("Training fertig. Q-Tabelle in 'q_table.npy' gespeichert.")
