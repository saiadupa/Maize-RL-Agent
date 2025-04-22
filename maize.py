import pygame
import random
import numpy as np
import os
import pandas as pd
from collections import deque

size = 10
width = size * 50
height = size * 50
win = pygame.display.set_mode((width, height))
fps = 30

actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((size * size, 4))

# Learning parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0001
alpha = 0.7
gamma = 0.95
q_table_file = 'q_table.csv'

# Ensure goal is reachable using BFS
def is_goal_reachable(grid):
    start = (0, 0)
    goal = (size - 1, size - 1)
    queue = deque([start])
    visited = set([start])
    
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] != 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False

# Generate grid until goal is reachable
def create_grid(size):
    while True:
        grid = np.random.choice([0, 1], (size, size), p=[0.1, 0.9])  
        grid[0][0] = 1  # Start point
        grid[size - 1][size - 1] = 2  # End point (goal)
        if is_goal_reachable(grid):
            return grid

grid = create_grid(size)

def get_epsilon(episode):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

def draw_window(win, grid):
    win.fill((0, 0, 0))
    for i in range(size):
        for j in range(size):
            if grid[j][i] == 1:
                color = (255, 255, 255)  # Open path
            elif grid[j][i] == 0:
                color = (255, 0, 0)  # Obstacle
            elif grid[j][i] == 2:
                color = (0, 255, 0)  # Goal
            pygame.draw.rect(win, color, (i * 50, j * 50, 48, 48), 0)

def take_action(state, epsilon):
    if random.random() > epsilon:
        return actions[np.argmax(q_table[state])]
    else:
        possible_actions = []
        if state >= size:
            possible_actions.append('up')
        if state < size * (size - 1):
            possible_actions.append('down')
        if state % size != 0:
            possible_actions.append('left')
        if state % size != size - 1:
            possible_actions.append('right')
        return random.choice(possible_actions)

def update_q_table(state, action, reward, next_state):
    q_table[state][actions.index(action)] += alpha * (
        reward + gamma * np.max(q_table[next_state]) - q_table[state][actions.index(action)]
    )

# ✅ Draw best path using Q-table
def draw_best_path():
    current_pos = (0, 0)  # Start at top left
    path = [current_pos]
    
    for _ in range(size * size):
        state = current_pos[1] * size + current_pos[0]
        action = np.argmax(q_table[state])

        if action == 0 and current_pos[1] > 0:   # Up
            next_pos = (current_pos[0], current_pos[1] - 1)
        elif action == 1 and current_pos[1] < size - 1:  # Down
            next_pos = (current_pos[0], current_pos[1] + 1)
        elif action == 2 and current_pos[0] > 0:  # Left
            next_pos = (current_pos[0] - 1, current_pos[1])
        elif action == 3 and current_pos[0] < size - 1:  # Right
            next_pos = (current_pos[0] + 1, current_pos[1])
        else:
            break

        if grid[next_pos[1]][next_pos[0]] == 0:
            break

        path.append(next_pos)
        current_pos = next_pos

        if current_pos == (size - 1, size - 1):
            break

    for i in range(len(path) - 1):
        start = (path[i][0] * 50 + 25, path[i][1] * 50 + 25)
        end = (path[i + 1][0] * 50 + 25, path[i + 1][1] * 50 + 25)
        pygame.draw.line(win, (0, 255, 255), start, end, 5)

def save_q_table():
    df = pd.DataFrame(q_table)
    df.to_csv(q_table_file, index=False)

def load_q_table():
    global q_table
    if os.path.exists(q_table_file):
        df = pd.read_csv(q_table_file)
        q_table = df.to_numpy()
        print(f"Q-table loaded from '{q_table_file}'")

def main():
    load_q_table()

    x_pos, y_pos = 25, 25
    clock = pygame.time.Clock()
    episode = 0
    state = 0
    steps = 0
    wins = 0
    max_episodes = 5000

    while episode < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q_table()
                pygame.quit()
                return
        clock.tick(fps)
        epsilon = get_epsilon(episode)

        draw_window(win, grid)
        pygame.draw.circle(win, (0, 0, 255), (x_pos, y_pos), 20, 0)
        draw_best_path()

        action = take_action(state, epsilon)

        new_x, new_y = x_pos, y_pos
        if action == 'left' and x_pos > 25:
            new_x -= 50
        elif action == 'right' and x_pos < width - 25:
            new_x += 50
        elif action == 'up' and y_pos > 25:
            new_y -= 50
        elif action == 'down' and y_pos < height - 25:
            new_y += 50

        next_state = size * (new_y // 50) + (new_x // 50)

        if grid[new_y // 50][new_x // 50] != 0:
            x_pos, y_pos = new_x, new_y
            state = next_state

        steps += 1
        update_q_table(state, action, 0, next_state)

        if state == size * size - 1:
            update_q_table(state, action, 100, next_state)  # Reward for success
            wins += 1
            print(f"✅ Episode {episode + 1}: REACHED GOAL ✅")
            episode += 1
            steps = 0
            save_q_table()

        elif steps > 500:
            print(f"❌ Episode {episode + 1}: TIMED OUT ❌")
            episode += 1
            steps = 0
            save_q_table()


        pygame.display.flip()
        pygame.time.delay(50)

    save_q_table()
    print(f"🏆 Total Wins: {wins}/{episode}")

if __name__ == "__main__":
    main()