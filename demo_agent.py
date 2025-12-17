# demo_agent.py
import time
import numpy as np
from princess_env import PrincessEnv

def run_demo(num_episodes=3, sleep_time=0.3):
    # Gelernte Q-Tabelle laden
    Q = np.load("q_table.npy")
    env = PrincessEnv()

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        print(f"\n=== Demo Episode {ep+1} ===")

        env.render()
        while not done:
            # Greedy Policy: immer die Aktion mit dem höchsten Q-Wert
            action = int(np.argmax(Q[state]))

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

            env.render()
            time.sleep(sleep_time)

        print(f"Episode {ep+1} Reward: {total_reward:.2f}, Info: {info}")

if __name__ == "__main__":
    run_demo()
