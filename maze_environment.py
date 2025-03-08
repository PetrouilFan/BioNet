import numpy as np
import pygame
import random
import time

class MazeEnvironment:
    """
    A simple maze environment with walls and a target
    The player can move in 4 directions: up, down, left, right
    Visualization using pygame instead of matplotlib
    """
    def __init__(self, width=10, height=10, wall_density=0.3, cell_size=30):
        self.width = width
        self.height = height
        self.maze = np.zeros((height, width))  # 0 = empty, 1 = wall
        self.player_pos = [1, 1]  # Start at top-left corner
        self.target_pos = [height-2, width-2]  # Target at bottom-right corner
        self.steps = 0
        self.max_steps = width * height * 2  # Maximum steps before reset
        
        # Generate maze with random walls
        self._generate_maze(wall_density)
        
        # History for visualization
        self.path_history = [self.player_pos.copy()]

        # For pygame visualization
        self.cell_size = cell_size
        self.screen_width = (width * cell_size) + 300  # Extra space for neural info
        self.screen_height = height * cell_size
        self.screen = None
        self.pygame_initialized = False

        # Store perception and output values for display
        self.perception_values = [0] * 8
        self.output_values = [0] * 4
        
    def _generate_maze(self, wall_density):
        """Generate a random maze with walls"""
        # Add border walls
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Add random walls
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if random.random() < wall_density:
                    # Don't place walls at start or target positions
                    if not ((y == self.player_pos[0] and x == self.player_pos[1]) or 
                            (y == self.target_pos[0] and x == self.target_pos[1])):
                        self.maze[y, x] = 1
        
        # Ensure there's a path from start to target (simple method)
        self.maze[1:self.height-1, 1] = 0  # Clear path down
        self.maze[self.height-2, 1:self.width-1] = 0  # Clear path right
        
    def reset(self):
        """Reset the player to starting position"""
        self.player_pos = [1, 1]
        self.steps = 0
        self.path_history = [self.player_pos.copy()]
        return self.get_observation()
    
    def get_observation(self):
        """Get observations of the 8 surrounding squares"""
        y, x = self.player_pos
        # Check all 8 adjacent squares (clockwise from top)
        surroundings = [
            self.maze[y-1, x],    # Up
            self.maze[y-1, x+1],  # Up-Right 
            self.maze[y, x+1],    # Right
            self.maze[y+1, x+1],  # Down-Right
            self.maze[y+1, x],    # Down
            self.maze[y+1, x-1],  # Down-Left
            self.maze[y, x-1],    # Left
            self.maze[y-1, x-1]   # Up-Left
        ]
        
        # Store for visualization
        self.perception_values = surroundings
        return surroundings
    
    def step(self, action):
        """Take action and return reward and new observation"""
        # action: 0=up, 1=right, 2=down, 3=left
        self.steps += 1
        
        # Determine new position based on action
        new_pos = self.player_pos.copy()
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Right
            new_pos[1] += 1
        elif action == 2:  # Down
            new_pos[0] += 1
        elif action == 3:  # Left
            new_pos[1] -= 1
        
        # Check if the move is valid (not hitting a wall)
        reward = -0.01  # Small negative reward for each step
        reached_target = False
        
        if self.maze[new_pos[0], new_pos[1]] == 1:
            # Hit a wall, stay in place
            reward = -0.5  # Penalty for hitting wall
        else:
            # Valid move, update position
            self.player_pos = new_pos
            self.path_history.append(self.player_pos.copy())
            
            # Check if reached target
            if self.player_pos[0] == self.target_pos[0] and self.player_pos[1] == self.target_pos[1]:
                reward = 1.0  # Bonus for reaching target
                reached_target = True
                
            # Add reward based on distance to target
            prev_distance = abs(self.path_history[-2][0] - self.target_pos[0]) + abs(self.path_history[-2][1] - self.target_pos[1])
            curr_distance = abs(self.player_pos[0] - self.target_pos[0]) + abs(self.player_pos[1] - self.target_pos[1])
            reward += 0.05 * (prev_distance - curr_distance)
        
        # Get new observation
        observation = self.get_observation()
        
        # Check for episode termination
        done = reached_target or self.steps >= self.max_steps
        
        return observation, reward, done

    def init_pygame(self):
        """Initialize pygame for visualization"""
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Maze Navigation")
            self.pygame_initialized = True
            self.font = pygame.font.Font(None, 24)

    def close_pygame(self):
        """Close pygame"""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

    def render(self):
        """Visualize the current state of the maze using pygame"""
        self.init_pygame()
        
        # Clear screen
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw maze
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                
                # Draw cell based on content
                if self.maze[y, x] == 1:
                    # Wall
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Black
                else:
                    # Path
                    pygame.draw.rect(self.screen, (220, 220, 220), rect)  # Light gray
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Border
        
        # Draw path history
        for pos in self.path_history[:-1]:  # Exclude current position
            center_x = pos[1] * self.cell_size + self.cell_size // 2
            center_y = pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, (100, 100, 250), (center_x, center_y), self.cell_size // 5)
                    
        # Draw target
        target_center_x = self.target_pos[1] * self.cell_size + self.cell_size // 2
        target_center_y = self.target_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, (0, 255, 0), (target_center_x, target_center_y), self.cell_size // 2)
        
        # Draw player
        player_center_x = self.player_pos[1] * self.cell_size + self.cell_size // 2
        player_center_y = self.player_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), (player_center_x, player_center_y), self.cell_size // 2)
        
        # Draw perception and output values in the right sidebar
        sidebar_x = self.width * self.cell_size + 10
        
        # Draw heading
        steps_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
        self.screen.blit(steps_text, (sidebar_x, 10))
        
        # Draw perception values (8 inputs)
        perception_heading = self.font.render("Perception (8 inputs):", True, (0, 0, 0))
        self.screen.blit(perception_heading, (sidebar_x, 50))
        
        directions = ["Up", "Up-Right", "Right", "Down-Right", "Down", "Down-Left", "Left", "Up-Left"]
        for i, (value, direction) in enumerate(zip(self.perception_values, directions)):
            y_pos = 80 + i * 25
            # Draw bar to represent value
            bar_width = int(float(value) * 100)
            pygame.draw.rect(self.screen, (0, 0, 200), (sidebar_x, y_pos, bar_width, 15))
            # Draw label
            text = self.font.render(f"{direction}: {float(value):.1f}", True, (0, 0, 0))
            self.screen.blit(text, (sidebar_x + 110, y_pos))
        
        # Draw output values (4 directions)
        output_heading = self.font.render("Neural Outputs (4 directions):", True, (0, 0, 0))
        self.screen.blit(output_heading, (sidebar_x, 300))
        
        action_names = ["Up", "Right", "Down", "Left"]
        for i, (value, action) in enumerate(zip(self.output_values, action_names)):
            y_pos = 330 + i * 25
            # Draw bar to represent value (normalize if needed)
            if isinstance(value, (int, float)):
                normalized_value = min(max(float(value), 0), 1)
                bar_width = int(normalized_value * 100)
            else:
                bar_width = 0
            
            pygame.draw.rect(self.screen, (200, 0, 0), (sidebar_x, y_pos, bar_width, 15))
            # Draw label
            text = self.font.render(f"{action}: {float(value):.2f}", True, (0, 0, 0))
            self.screen.blit(text, (sidebar_x + 110, y_pos))
        
        # Update display
        pygame.display.flip()
        
        # Process pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_pygame()
                return False
        
        # Small delay to make visualization smoother
        time.sleep(0.1)
        return True
