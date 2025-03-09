import random
import pygame
import math
import numpy as np
from collections import deque
import torch

GRID_SIZE = 10
WALL = 1
FREE = 0

# Colors for visualization
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (148, 0, 211)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

SYNAPSE_VISUAL_WEIGHT_DIVISOR = 10.0
SYNAPSE_VISUAL_WIDTH_SCALE = 5

def get_percepts(maze, x, y, exit_pos=None):
    percepts = []
    directions = [
        (0, -1),   # Up
        (0, 1),    # Down
        (-1, 0),   # Left
        (1, 0),    # Right
        (1, -1),   # Up-right
        (1, 1),    # Down-right
        (-1, -1),  # Up-left
        (-1, 1)    # Down-left
    ]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if maze[ny][nx] == WALL:
            percepts.append(-1)  # Wall
        elif exit_pos and (nx, ny) == exit_pos:
            percepts.append(1)   # Exit
        else:
            percepts.append(0)   # Free space
            
    return percepts

def generate_maze(width, height, wall_density=0.3):
    """Generate a maze with internal walls based on wall_density"""
    # Start with all cells as FREE spaces
    maze = [[FREE for _ in range(width)] for _ in range(height)]
    
    # Ensure border cells are all walls
    for x in range(width):
        maze[0][x] = WALL
        maze[height-1][x] = WALL
    for y in range(height):
        maze[y][0] = WALL
        maze[y][width-1] = WALL
    
    # Add random internal walls based on wall_density
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Skip start and exit positions
            if (x == 1 and y == 1) or (x == width-2 and y == height-2):
                continue
            # Randomly place walls
            if random.random() < wall_density:
                maze[y][x] = WALL
    
    return maze

def has_path(maze, start, exit):
    """
    Use BFS to check if there's a path from start to exit in the maze.
    """
    queue = deque([start])
    visited = set([start])
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == exit:
            return True
            
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and 
                maze[ny][nx] == FREE and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))
                
    return False

class MazeGame:
    def __init__(self, width=21, height=21, cell_size=80):
        # Store game dimensions
        self.maze_width = width
        self.maze_height = height
        self.start = (1, 1)
        self.exit = (width - 2, height - 2)
        
        # Generate maze and ensure path exists
        self.maze = self.generate_maze_with_path()
        
        self.position = self.start
        self.brain = None  # To be set externally
        # Add step counter and max steps calculation
        self.step_counter = 0
        self.estimated_min_steps = self.calculate_manhattan_distance(self.start, self.exit)
        self.max_allowed_steps = self.maze_width * self.maze_height  # Worst case scenario

    def generate_maze_with_path(self):
        """Generate a maze and ensure there's a path from start to exit."""
        maze = generate_maze(self.maze_width, self.maze_height)
        
        # Ensure start and exit are free cells
        start_x, start_y = self.start
        exit_x, exit_y = self.exit
        maze[start_y][start_x] = FREE
        maze[exit_y][exit_x] = FREE
        
        # Verify the path exists
        if not has_path(maze, self.start, self.exit):
            # Recursively try again if no path exists (shouldn't happen often with Wilson's algorithm)
            return self.generate_maze_with_path()
        
        return maze
    
    def calculate_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_state(self):
        x, y = self.position
        return get_percepts(self.maze, x, y, self.exit)

    def step(self, action):
        # Increment step counter
        self.step_counter += 1
        
        # Map action to movement: 0=up, 1=down, 2=left, 3=right
        move = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}.get(action, (0, 0))
        x, y = self.position
        new_x = x + move[0]
        new_y = y + move[1]
        
        # Calculate distances for reward calculation
        old_distance = self.calculate_manhattan_distance(self.position, self.exit)
        new_distance = self.calculate_manhattan_distance((new_x, new_y), self.exit)
        
        # Check if new position is a wall
        if self.maze[new_y][new_x] == WALL:
            reward = -5  # Increased wall penalty
        else:
            self.position = (new_x, new_y)
            
            if self.position == self.exit:
                # Significantly increased exit reward
                efficiency_factor = self.estimated_min_steps / max(self.step_counter, 1)
                # Cap efficiency factor between 0.3 and 1.0
                efficiency_factor = min(1.0, max(0.3, efficiency_factor))
                reward = 50 + 50 * efficiency_factor  # 65-100 reward based on efficiency
            else:
                # Enhanced distance-based rewards
                if new_distance < old_distance:
                    reward = 1.0  # Increased reward for moving toward exit
                    # Add exploration bonus for first visit to a cell
                    if (new_x, new_y) not in self.visited_cells:
                        reward += 0.5
                else:
                    reward = -1  # Keep standard penalty for non-productive moves
                
                # Softer time pressure: reduced penalty and delayed onset
                grace_period = self.estimated_min_steps * 3  # Extended grace period
                if self.step_counter > grace_period:
                    step_penalty = -0.005 * (self.step_counter - grace_period)  # Reduced penalty coefficient
                    reward += step_penalty

            # Track visited cells for exploration bonus
            self.visited_cells.add((new_x, new_y))

        # Softer multiple neuron firing penalty with early training grace period
        if self.brain and hasattr(self.brain, 'output_neurons'):
            firing_count = 0
            firing_threshold = -55
            for neuron in self.brain.output_neurons:
                if len(neuron.potential_history) > 0 and neuron.potential_history[-1] > firing_threshold:
                    firing_count += 1
            # Only apply multiple firing penalty after some episodes
            if firing_count > 1 and hasattr(self.brain, 'episode_count') and self.brain.episode_count > 50:
                reward -= 1  # Reduced penalty

        done = self.position == self.exit
        return self.get_state(), reward, done

    def reset(self):
        self.maze = self.generate_maze_with_path()
        self.position = self.start
        self.exit = (self.maze_width - 2, self.maze_height - 2)
        # Reset step counter on new episode
        self.step_counter = 0
        self.estimated_min_steps = self.calculate_manhattan_distance(self.start, self.exit)
        # Initialize visited cells set for exploration tracking
        self.visited_cells = set([self.start])
        return self.get_state()

    def set_brain(self, brain):
        self.brain = brain

class MazeVisualization:
    def __init__(self, game: MazeGame):
        self.game = game
        # Fixed display dimensions
        self.display_width = 1280
        self.display_height = 960
        maze_panel_width = int(self.display_width * 0.30)   # ≈384px
        self.cell_size = int(maze_panel_width / self.game.maze_width)
        self.width = maze_panel_width
        self.nn_width = self.display_width - maze_panel_width
        self.nn_height = self.display_height
        self.nn_margin = 50
        self.firing_synapses = deque(maxlen=20)
        # Initialize pygame with fixed window size
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Neural Maze Environment")
        self.font = pygame.font.SysFont("Arial", 12)
        self.clock = pygame.time.Clock()
        self.show_visualization = True
        self.frame_delay = 100  # milliseconds between updates

    def visualize(self):
        if not pygame.get_init():
            return
        self.screen.fill(WHITE)
        self.draw_maze()
        if self.game.brain is not None:
            self.draw_neural_network()
        pygame.display.flip()
        # Process events and control frame rate
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); exit()
                elif event.key == pygame.K_v:
                    self.show_visualization = not self.show_visualization
                elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                    self.frame_delay = max(10, self.frame_delay - 10)
                elif event.key == pygame.K_MINUS:
                    self.frame_delay += 10
        self.clock.tick(1000 / self.frame_delay)

    def draw_maze(self):
        for y in range(len(self.game.maze)):
            for x in range(len(self.game.maze[y])):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                if self.game.maze[y][x] == WALL:
                    pygame.draw.rect(self.screen, BLACK, rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                    pygame.draw.rect(self.screen, GRAY, rect, 1)
                if (x, y) == self.game.exit:
                    # Reduced inflate so the exit is visible even in small cells
                    pygame.draw.rect(self.screen, GREEN, rect.inflate(-4, -4))
        # Draw agent
        x, y = self.game.position
        agent_rect = pygame.Rect(
            x * self.cell_size + self.cell_size * 0.25,
            y * self.cell_size + self.cell_size * 0.25,
            self.cell_size * 0.5,
            self.cell_size * 0.5
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect)
        percepts = self.game.get_state()
        percept_positions = [
            (x, y-1), (x, y+1), (x-1, y), (x+1, y),
            (x+1, y-1), (x+1, y+1), (x-1, y-1), (x-1, y+1)
        ]
        for i, (px, py) in enumerate(percept_positions):
            if px < 0 or py < 0 or px >= len(self.game.maze[0]) or py >= len(self.game.maze):
                continue
            indicator_rect = pygame.Rect(
                px * self.cell_size + self.cell_size * 0.4,
                py * self.cell_size + self.cell_size * 0.4,
                self.cell_size * 0.2,
                self.cell_size * 0.2
            )
            color = RED if percepts[i] == -1 else GREEN if percepts[i] == 1 else YELLOW
            pygame.draw.rect(self.screen, color, indicator_rect)
            text = self.font.render(f"{percepts[i]:.1f}", True, BLACK)
            text_rect = text.get_rect(center=indicator_rect.center)
            self.screen.blit(text, text_rect)

    def draw_neural_network(self):
        nn_rect = pygame.Rect(self.width, 0, self.nn_width, self.nn_height)
        pygame.draw.rect(self.screen, (240, 240, 240), nn_rect)
        pygame.draw.rect(self.screen, BLACK, nn_rect, 2)
        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_text = title_font.render("Neural Network - Real-time Activity", True, BLACK)
        self.screen.blit(title_text, (self.width + 10, 10))
        input_size = len(self.game.brain.input_neurons)
        hidden_size = len(self.game.brain.hidden_neurons)
        output_size = len(self.game.brain.output_neurons)
        module_centers = {
            'input': (self.width + self.nn_width * 0.25, self.nn_height * 0.5),
            'hidden': (self.width + self.nn_width * 0.5, self.nn_height * 0.5),
            'output': (self.width + self.nn_width * 0.75, self.nn_height * 0.5)
        }
        module_radii = {
            'input': min(200, max(80, input_size * 15)),
            'hidden': min(200, max(80, hidden_size * 15)),
            'output': min(200, max(80, output_size * 15))
        }
        self.draw_module_backgrounds(module_centers, module_radii)
        input_positions = self.calculate_circular_positions(module_centers['input'], module_radii['input'] * 0.7, input_size)
        hidden_positions = self.calculate_circular_positions(module_centers['hidden'], module_radii['hidden'] * 0.7, hidden_size)
        output_positions = self.calculate_circular_positions(module_centers['output'], module_radii['output'] * 0.7, output_size)
        self.draw_circular_synapses(input_positions, hidden_positions, output_positions)
        for i, pos in enumerate(input_positions):
            value = self.game.brain.input_neurons[i].membrane_potential
            norm_value = min(1.0, max(0, (value + 65.0) / 66.0))
            percepts = self.game.get_state()
            percept_value = percepts[i] if i < len(percepts) else "N/A"
            self.draw_neuron(pos, value, norm_value, f"In {i}", i < 4, 
                            extra_info=[f"Percept: {percept_value}", f"MP: {value:.2f}mV"])
        for i, pos in enumerate(hidden_positions):
            value = self.game.brain.hidden_neurons[i].membrane_potential
            norm_value = min(1.0, max(0, (value + 65.0) / 66.0))
            self.draw_neuron(pos, value, norm_value, f"H{i}", False,
                           extra_info=[f"MP: {value:.2f}mV"])
        for i, pos in enumerate(output_positions):
            value = self.game.brain.output_neurons[i].membrane_potential
            norm_value = min(1.0, max(0, (value + 65.0) / 66.0))
            highlight = (self.game.brain.last_action == i)
            direction_labels = ["Up", "Down", "Left", "Right"]
            label = direction_labels[i] if i < len(direction_labels) else f"Out {i}"
            action_info = "ACTIVE" if highlight else ""
            self.draw_neuron(pos, value, norm_value, label, highlight,
                            extra_info=[f"MP: {value:.2f}mV", action_info])
        self.draw_neural_network_legend(self.width + 10, self.nn_height - 90)

    def draw_module_backgrounds(self, module_centers, module_radii):
        module_colors = {'input': (200, 200, 255, 100), 'hidden': (220, 255, 220, 100), 'output': (255, 220, 220, 100)}
        for module_name, center in module_centers.items():
            radius = module_radii[module_name]
            surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, module_colors[module_name], (radius, radius), radius)
            self.screen.blit(surf, (center[0]-radius, center[1]-radius))
            pygame.draw.circle(self.screen, BLACK, (int(center[0]), int(center[1])), radius, 2)
            label_font = pygame.font.SysFont("Arial", 18, bold=True)
            module_label = module_name.capitalize() + " Module"
            label_text = label_font.render(module_label, True, BLACK)
            self.screen.blit(label_text, (center[0] - label_text.get_width()//2, center[1] - radius - 30))

    def calculate_circular_positions(self, center, radius, count):
        positions = []
        if count == 0:
            return positions
        if count == 1:
            return [center]
        for i in range(count):
            angle = (2 * math.pi * i / count) - (math.pi / 2)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def draw_circular_synapses(self, input_positions, hidden_positions, output_positions):
        if not self.game.brain or not self.game.brain.synapses:
            return
        for key, synapse in self.game.brain.synapses.items():
            source_layer, source_idx, target_layer, target_idx = key
            weight = synapse.weight.item() if hasattr(synapse.weight, 'item') else float(synapse.weight)
            source_pos = None
            target_pos = None
            if source_layer == 'input' and source_idx < len(input_positions):
                source_pos = input_positions[source_idx]
            elif source_layer == 'hidden' and source_idx < len(hidden_positions):
                source_pos = hidden_positions[source_idx]
            if target_layer == 'hidden' and target_idx < len(hidden_positions):
                target_pos = hidden_positions[target_idx]
            elif target_layer == 'output' and target_idx < len(output_positions):
                target_pos = output_positions[target_idx]
            if source_pos and target_pos:
                norm_weight = abs(weight) / SYNAPSE_VISUAL_WEIGHT_DIVISOR
                norm_weight = max(0.1, min(1.0, norm_weight))
                width = max(1, int(norm_weight * SYNAPSE_VISUAL_WIDTH_SCALE))
                color = (0, int(200 * norm_weight), 0) if weight >= 0 else (int(200 * norm_weight), 0, 0)
                if key in self.firing_synapses:
                    color = YELLOW
                    width += 1
                dx = target_pos[0] - source_pos[0]
                dy = target_pos[1] - source_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                offset_mag = min(80, max(30, dist * 0.3))
                offset = random.uniform(0.7, 1.3) * offset_mag
                perp_x = -dy / dist if dist > 0 else 0
                perp_y = dx / dist if dist > 0 else 0
                control_point = ((source_pos[0] + target_pos[0]) / 2 + perp_x * offset,
                                 (source_pos[1] + target_pos[1]) / 2 + perp_y * offset)
                points = []
                steps = 30
                for i in range(steps + 1):
                    t = i / steps
                    x = (1-t)**2 * source_pos[0] + 2*(1-t)*t * control_point[0] + t**2 * target_pos[0]
                    y = (1-t)**2 * source_pos[1] + 2*(1-t)*t * control_point[1] + t**2 * target_pos[1]
                    points.append((x, y))
                for i in range(len(points) - 1):
                    pygame.draw.line(self.screen, color, points[i], points[i+1], width)
                mid_point = points[len(points)//2]
                weight_font = pygame.font.SysFont("Arial", 10)
                weight_text = weight_font.render(f"{weight:.2f}", True, BLACK)
                text_bg = pygame.Rect(mid_point[0] - weight_text.get_width()//2 - 2,
                                      mid_point[1] - weight_text.get_height()//2 - 1,
                                      weight_text.get_width() + 4,
                                      weight_text.get_height() + 2)
                pygame.draw.rect(self.screen, WHITE, text_bg)
                pygame.draw.rect(self.screen, (200, 200, 200), text_bg, 1)
                self.screen.blit(weight_text, (mid_point[0] - weight_text.get_width()//2,
                                               mid_point[1] - weight_text.get_height()//2))
                if len(points) > 2:
                    self.draw_arrow_head(points[-1], points[-2], color, width)

    def draw_arrow_head(self, pos, prev_pos, color, width):
        dx = pos[0] - prev_pos[0]
        dy = pos[1] - prev_pos[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.0001:
            return
        dx, dy = dx/length, dy/length
        px, py = -dy, dx
        arrow_size = width + 4
        p1 = (pos[0] - dx*arrow_size - px*arrow_size/2,
              pos[1] - dy*arrow_size - py*arrow_size/2)
        p2 = (pos[0] - dx*arrow_size + px*arrow_size/2,
              pos[1] - dy*arrow_size + py*arrow_size/2)
        pygame.draw.polygon(self.screen, color, [pos, p1, p2])

    def draw_neural_network_legend(self, x, y):
        legend_font = pygame.font.SysFont("Arial", 12)
        legend_rect = pygame.Rect(x, y, self.nn_width - 20, 80)
        pygame.draw.rect(self.screen, (245, 245, 245), legend_rect)
        pygame.draw.rect(self.screen, BLACK, legend_rect, 1)
        title = legend_font.render("Legend:", True, BLACK)
        self.screen.blit(title, (x + 5, y + 5))
        pygame.draw.circle(self.screen, (0, 0, 255), (x + 15, y + 25), 7)
        pygame.draw.circle(self.screen, (255, 0, 0), (x + 15, y + 45), 7)
        inactive_text = legend_font.render("Inactive neuron (-70mV)", True, BLACK)
        active_text = legend_font.render("Active neuron (-55mV+)", True, BLACK)
        self.screen.blit(inactive_text, (x + 30, y + 20))
        self.screen.blit(active_text, (x + 30, y + 40))
        pygame.draw.line(self.screen, (0, 200, 0), (x + 200, y + 25), (x + 230, y + 25), 3)
        pygame.draw.line(self.screen, (200, 0, 0), (x + 200, y + 45), (x + 230, y + 45), 3)
        positive_text = legend_font.render("Excitatory synapse (positive weight)", True, BLACK)
        negative_text = legend_font.render("Inhibitory synapse (negative weight)", True, BLACK)
        self.screen.blit(positive_text, (x + 240, y + 20))
        self.screen.blit(negative_text, (x + 240, y + 40))
        pygame.draw.line(self.screen, YELLOW, (x + 200, y + 65), (x + 230, y + 65), 4)
        active_synapse_text = legend_font.render("Recently fired synapse", True, BLACK)
        self.screen.blit(active_synapse_text, (x + 240, y + 60))

    def draw_neuron(self, pos, value, norm_value, label=None, highlight=False, extra_info=None):
        x, y = pos
        radius = 12
        if isinstance(value, torch.Tensor):
            try:
                value = value.item()
            except:
                value = value.mean().item() if value.numel() > 0 else 0.0
        color = (
            int(255 * norm_value),
            50,
            int(255 * (1.0 - norm_value))
        )
        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        border_width = 3 if highlight else 1
        border_color = ORANGE if highlight else BLACK
        pygame.draw.circle(self.screen, border_color, (int(x), int(y)), radius, border_width)
        if label:
            label_font = pygame.font.SysFont("Arial", 11, bold=True)
            text = label_font.render(label, True, BLACK)
            self.screen.blit(text, (x - text.get_width() // 2, y - radius - 20))
        if extra_info:
            info_font = pygame.font.SysFont("Arial", 10)
            for i, info in enumerate(extra_info):
                if info:
                    text = info_font.render(info, True, BLACK)
                    text_bg = pygame.Rect(
                        x - text.get_width() // 2 - 2,
                        y + radius + 2 + i * 15,
                        text.get_width() + 4,
                        text.get_height()
                    )
                    pygame.draw.rect(self.screen, (240, 240, 240), text_bg)
                    pygame.draw.rect(self.screen, (200, 200, 200), text_bg, 1)
                    self.screen.blit(text, (x - text.get_width() // 2, y + radius + 2 + i * 15))
        threshold = self.game.brain.neuron_kwargs['threshold'] if self.game.brain else -55
        rest = self.game.brain.neuron_kwargs['rest_potential'] if self.game.brain else -70
        total_range = threshold - rest
        if total_range > 0:
            threshold_y = y - int((threshold - rest) / total_range * radius * 0.8)
            pygame.draw.line(self.screen, (50, 50, 50), 
                            (x - radius + 2, threshold_y), 
                            (x + radius - 2, threshold_y), 1)
