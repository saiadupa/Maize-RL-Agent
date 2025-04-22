# Q-Learning Agent for 2D Maze Navigation

This project implements a reinforcement learning agent using Q-learning to solve randomly generated 2D mazes. The environment is rendered in real-time using Pygame, and the agent learns to reach the goal from the start position by avoiding obstacles.

## Features

- Custom maze generation with solvability validation (BFS)
- Q-learning algorithm with tabular Q-table
- Epsilon-greedy action selection with exponential decay
- Real-time Pygame visualization of agent and learned path
- Persistent Q-table using CSV for continued learning

## How It Works

1. A 10×10 grid is randomly generated with obstacles and a goal.
2. The agent starts at the top-left corner and learns through interaction.
3. The Q-table is updated based on rewards (goal = 1, others = 0).
4. Over episodes, the agent improves its navigation strategy.
5. The best-known path is rendered using the Q-table.

## Requirements

- Python 3.7+
- `pygame`
- `numpy`
- `pandas`

Install dependencies:
```bash
pip install pygame numpy pandas
```

## Key Hyperparameters

- Learning Rate (α): 0.7
- Discount Factor (γ): 0.95
- Exploration Decay Rate: 0.0001
- Max Episodes: 5000
- Max Steps per Episode: 500

## Files

- `main.py`: Main Q-learning training script
- `q_table.csv`: Saved Q-table (auto-loaded and updated)

## Output

- The agent is visualized as a blue circle.
- Obstacles (red), paths (white), goal (green), best path (cyan).

## Evaluation Metrics

| Metric               | Value          |
|----------------------|----------------|
| Grid Size            | 10 × 10        |
| Total Episodes       | 5000           |
| Success Rate         | ~84.76%        |
| Avg Steps to Goal    | ~87.4          |
| Q-table Persistence  | CSV            |
