# Q-Learning Cliff Walking

A Reinforcement Learning project implementing the **Q-Learning algorithm** to solve the classic **Cliff Walking** environment using Python and Gymnasium.

This project demonstrates how an agent learns an optimal policy through trial and error using **off-policy Temporal Difference (TD) learning**.

---

## 📌 Project Overview

Cliff Walking is a classic Reinforcement Learning problem where an agent must navigate from a starting position to a goal while avoiding a dangerous cliff.

The agent receives rewards based on its actions and gradually learns the optimal path by interacting with the environment over multiple episodes.

This project uses the **Q-Learning algorithm**, a model-free and off-policy reinforcement learning technique, to train the agent.

---

## 🧠 What is Q-Learning?

Q-Learning is a **model-free, off-policy reinforcement learning algorithm** that learns the optimal action-value function.

The algorithm estimates the expected future rewards for each state-action pair and gradually updates these values through interaction with the environment.

The Q-Learning update rule is:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]
```

Where:

* `Q(s, a)` = Current Q-value for a state-action pair
* `α` = Learning rate
* `r` = Reward received after taking an action
* `γ` = Discount factor
* `s'` = Next state
* `max Q(s', a')` = Maximum possible Q-value in the next state

---

## 🎯 Objective

The objective of this project is to train an agent that can:

* Navigate from the starting position to the goal.
* Avoid falling into the cliff.
* Maximize cumulative rewards.
* Learn an optimal path through repeated interactions.
* Balance exploration and exploitation using an epsilon-greedy policy.

---

## 🗺️ Cliff Walking Environment

The environment is a grid-based world consisting of:

* **Start:** Bottom-left corner of the grid.
* **Goal:** Bottom-right corner of the grid.
* **Cliff:** Dangerous cells located between the start and goal.
* **Agent:** Learns to navigate safely to the goal.

If the agent falls into the cliff, it receives a large negative reward and is returned to the starting position.

The environment used in this project is:

```python
env = gym.make("CliffWalking-v1")
```

---

## ⚙️ Technologies Used

* Python
* Jupyter Notebook
* Gymnasium
* NumPy
* Matplotlib

---

## 📂 Project Structure

```text
Q-Learning-Cliff-Walking/
│
├── Q_Learning_Cliff_Walking.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Q-Learning-Cliff-Walking.git
```

### 2. Navigate to the Project Directory

```bash
cd Q-Learning-Cliff-Walking
```

### 3. Install Dependencies

```bash
pip install gymnasium numpy matplotlib jupyter
```

---

## ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the following notebook:

```text
Q_Learning_Cliff_Walking.ipynb
```

Run all cells sequentially to train the Q-Learning agent and evaluate the learned policy.

---

# 🔄 Q-Learning Algorithm Workflow

The Q-Learning algorithm follows these steps:

1. Initialize the Q-table with zeros.
2. Reset the environment.
3. Select an action using an epsilon-greedy policy.
4. Perform the selected action.
5. Observe the reward and next state.
6. Find the maximum Q-value for the next state.
7. Update the current Q-value using the Q-Learning update rule.
8. Move to the next state.
9. Repeat until the episode terminates.
10. Train the agent over multiple episodes.

---

## 🧮 Q-Table

The Q-table stores the expected reward for every state-action pair.

It is initialized as:

```python
Q = np.zeros(
    (
        env.observation_space.n,
        env.action_space.n
    )
)
```

Each row represents a state, and each column represents a possible action.

---

## 🎲 Epsilon-Greedy Policy

The agent uses an epsilon-greedy strategy to balance:

### Exploration

The agent selects a random action to discover new possibilities.

### Exploitation

The agent selects the action with the highest Q-value based on previous learning.

Example implementation:

```python
def epsilon_greedy(state):

    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[state])
```

---

## 📈 Q-Learning Update

The core update equation used in this project is:

```python
Q[state, action] += alpha * (
    reward
    + gamma * np.max(Q[next_state])
    - Q[state, action]
)
```

The agent continuously updates its Q-values until it learns an effective policy.

---

## 📊 Key Hyperparameters

| Hyperparameter      | Description                                   |
| ------------------- | --------------------------------------------- |
| Learning Rate (α)   | Controls how quickly the Q-values are updated |
| Discount Factor (γ) | Determines the importance of future rewards   |
| Epsilon (ε)         | Controls exploration versus exploitation      |
| Episodes            | Number of training iterations                 |

---

## 🧩 Concepts Demonstrated

This project demonstrates the following Reinforcement Learning concepts:

* Reinforcement Learning
* Markov Decision Process (MDP)
* Q-Learning
* Model-Free Learning
* Off-Policy Learning
* Temporal Difference Learning
* Q-Table
* Epsilon-Greedy Policy
* Exploration vs Exploitation
* Reward Maximization

---

## 🔍 Q-Learning vs SARSA

| Feature           | Q-Learning            | SARSA                   |
| ----------------- | --------------------- | ----------------------- |
| Learning Type     | Off-Policy            | On-Policy               |
| Next State Update | Maximum Q-value       | Next selected action    |
| Update Rule       | `max Q(s',a')`        | `Q(s',a')`              |
| Policy Learned    | Optimal greedy policy | Current behavior policy |

---

## 📈 Expected Results

During the initial training episodes, the agent explores the environment randomly and may frequently fall into the cliff.

As training progresses:

* Q-values are updated.
* The agent learns which actions produce better rewards.
* The number of unnecessary movements decreases.
* The agent discovers an efficient path to the goal.

After sufficient training, the agent follows the learned policy using the highest Q-value for each state.

---

## 🔮 Future Improvements

Possible extensions to this project include:

* Compare Q-Learning with SARSA.
* Visualize the learned optimal policy.
* Plot rewards across training episodes.
* Implement Expected SARSA.
* Experiment with different epsilon values.
* Implement epsilon decay.
* Compare different learning rates and discount factors.
* Extend the project using Deep Q-Networks (DQN).

---

## 📚 Learning Resources

* Reinforcement Learning: An Introduction by Sutton and Barto
* Gymnasium Documentation
* Temporal Difference Learning
* Q-Learning Algorithm

---

## 👨‍💻 Author

**Anjali Sonavane**

Aspiring AI and Machine Learning Engineer interested in:

* Machine Learning
* Deep Learning
* Reinforcement Learning
* Artificial Intelligence
* Cloud Computing

---

## ⭐ Support

If you found this project useful, consider giving the repository a star.

---



This project is created for educational and learning purposes.
