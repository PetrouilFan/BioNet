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

def generate_maze(width, height):
    """Generate a maze using Wilson's algorithm"""
    maze = [[WALL for _ in range(width)] for _ in range(height)]
    unvisited = [(x, y) for x in range(1, width-1) for y in range(1, height-1)]
    path = []

    def neighbors(x, y):
        return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= x + dx < width and 0 <= y + dy < height]

    def is_valid(x, y):
        return 0 <= x < width and 0 <= y < height and maze[y][x] == WALL

    start = random.choice(unvisited)
    unvisited.remove(start)
    maze[start[1]][start[0]] = FREE

    while unvisited:
        current = random.choice(unvisited)
        path.append(current)

        while current in unvisited:
            next_cell = random.choice(neighbors(*current))
            if next_cell in path:
                path = path[:path.index(next_cell) + 1]
            else:
                path.append(next_cell)
            current = next_cell

        for cell in path:
            if cell in unvisited:
                unvisited.remove(cell)
                maze[cell[1]][cell[0]] = FREE
        path = []

    return maze

class MazeEnvironment:
    def __init__(self, width=5, height=5, cell_size=80):
        # Store maze dimensions separately from display dimensions
        self.maze_width = width
        self.maze_height = height
        self.maze = generate_maze(width, height)
        self.start = (1, 1)
        self.exit = (width - 2, height - 2)
        self.position = self.start
        
        # Set fixed display dimensions
        total_display_width = 1280
        total_display_height = 960
        # Allocate 30% width for maze viz and 70% for neural network
        maze_panel_width = int(total_display_width * 0.30)   # ≈384px
        self.cell_size = int(maze_panel_width / self.maze_width)  # adjust cell size to fit maze panel
        self.width = maze_panel_width  # maze viz width
        
        self.nn_width = total_display_width - maze_panel_width  # neural network gets remaining 70%
        self.nn_height = total_display_height
        self.nn_margin = 50
     
        self.display_width = total_display_width
        self.display_height = total_display_height
        
        # Synapse visualization
        self.firing_synapses = deque(maxlen=20)  # Store recently fired synapses
        
        # Initialize pygame with fixed window size
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Neural Maze Environment")
        self.font = pygame.font.SysFont("Arial", 12)
        self.clock = pygame.time.Clock()
        
        # Visualization state
        self.show_visualization = True
        self.frame_delay = 100  # milliseconds between updates
        self.brain = None  # Will be set externally

    def reset(self):
        # Use maze_width and maze_height instead of width and height
        self.maze = generate_maze(self.maze_width, self.maze_height)
        self.position = self.start
        # Update exit position based on new maze
        self.exit = (self.maze_width - 2, self.maze_height - 2)
        
        # Visualize the maze immediately after generation
        if self.show_visualization:
            self.visualize()
            
        return self.get_state()

    def get_state(self):
        x, y = self.position
        return get_percepts(self.maze, x, y, self.exit)

    def step(self, action):
        # Map action to movement: 0=up, 1=down, 2=left, 3=right
        move = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}.get(action, (0, 0))
        x, y = self.position
        new_x = x + move[0]
        new_y = y + move[1]

        # Check if new position is a wall
        if self.maze[new_y][new_x] == WALL:
            reward = -2  # hit a wall: negative reward; position unchanged
        else:
            self.position = (new_x, new_y)
            if self.position == self.exit:
                reward = 10  # bonus for reaching the exit
            else:
                reward = -1
        
        # Penalize if multiple output neurons are firing
        if self.brain and hasattr(self.brain, 'output_neurons'):
            firing_count = 0
            firing_threshold = -55  # Typical firing threshold for neurons
            
            for neuron in self.brain.output_neurons:
                if len(neuron.potential_history) > 0 and neuron.potential_history[-1] > firing_threshold:
                    firing_count += 1
            
            if firing_count > 1:
                reward -= 2  # Apply penalty for multiple neurons firing

        done = self.position == self.exit
        
        # Update visualization
        if self.show_visualization:
            self.visualize()
            
        return self.get_state(), reward, done
    
    def set_brain(self, brain):
        """Set the brain for visualization purposes"""
        self.brain = brain
    
    def visualize(self):
        """Main visualization method that draws everything"""
        if not pygame.get_init():
            return
            
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw maze
        self.draw_maze()
        
        # Draw neural network if brain is available
        if self.brain is not None:
            self.draw_neural_network()
        
        # Update display
        pygame.display.flip()
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
                elif event.key == pygame.K_v:
                    self.show_visualization = not self.show_visualization
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.frame_delay = max(10, self.frame_delay - 10)
                elif event.key == pygame.K_MINUS:
                    self.frame_delay += 10
        
        # Control frame rate
        self.clock.tick(1000 / self.frame_delay)  # Convert to FPS
    
    def draw_maze(self):
        """Draw the maze grid and agent position"""
        # Draw cells
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                if self.maze[y][x] == WALL:
                    pygame.draw.rect(self.screen, BLACK, rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                    pygame.draw.rect(self.screen, GRAY, rect, 1)
                
                # Mark exit
                if (x, y) == self.exit:
                    pygame.draw.rect(self.screen, GREEN, rect.inflate(-20, -20))
        
        # Draw agent
        x, y = self.position
        agent_rect = pygame.Rect(
            x * self.cell_size + self.cell_size * 0.25,
            y * self.cell_size + self.cell_size * 0.25,
            self.cell_size * 0.5,
            self.cell_size * 0.5
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect)
        
        # Draw percepts as colored squares around agent
        percepts = self.get_state()
        percept_positions = [
            (x, y-1),  # Up
            (x, y+1),  # Down
            (x-1, y),  # Left
            (x+1, y),  # Right
            (x+1, y-1),  # Up-Right
            (x+1, y+1),  # Down-Right
            (x-1, y-1),  # Up-Left
            (x-1, y+1)   # Down-Left
        ]
        
        for i, (px, py) in enumerate(percept_positions):
            # Skip if outside maze boundaries
            if px < 0 or py < 0 or px >= len(self.maze[0]) or py >= len(self.maze):
                continue
                
            # Draw small indicator showing percept value
            percept_value = percepts[i]
            indicator_rect = pygame.Rect(
                px * self.cell_size + self.cell_size * 0.4,
                py * self.cell_size + self.cell_size * 0.4,
                self.cell_size * 0.2,
                self.cell_size * 0.2
            )
            
            # Color based on percept value
            if percept_value == -1:
                color = RED        # Wall
            elif percept_value == 1:
                color = GREEN      # Exit
            else:
                color = YELLOW     # Free space
                
            pygame.draw.rect(self.screen, color, indicator_rect)
            
            # Draw percept value
            text = self.font.render(f"{percept_value:.1f}", True, BLACK)
            text_rect = text.get_rect(center=indicator_rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_neural_network(self):
        """Draw the neural network visualization"""
        if self.brain is None:
            return
            
        # Neural network visualization area
        nn_rect = pygame.Rect(self.width, 0, self.nn_width, self.nn_height)
        pygame.draw.rect(self.screen, (240, 240, 240), nn_rect)
        pygame.draw.rect(self.screen, BLACK, nn_rect, 2)
        
        # Title for the neural network display
        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_text = title_font.render("Neural Network - Real-time Activity", True, BLACK)
        self.screen.blit(title_text, (self.width + 10, 10))
        
        # Get neuron counts from brain
        input_size = len(self.brain.input_neurons)
        hidden_size = len(self.brain.hidden_neurons)
        output_size = len(self.brain.output_neurons)
        
        # Calculate neuron positions
        input_spacing = self.nn_height / (input_size + 1)
        hidden_spacing = self.nn_height / (hidden_size + 1)
        output_spacing = self.nn_height / (output_size + 1)
        
        # Layer x-positions - distribute more evenly
        input_x = self.width + self.nn_margin + 50
        hidden_x = self.width + (self.nn_width / 2)
        output_x = self.width + self.nn_width - self.nn_margin - 50
        
        # Draw layer labels
        layer_font = pygame.font.SysFont("Arial", 14, bold=True)
        input_layer_text = layer_font.render("Input Layer", True, BLACK)
        hidden_layer_text = layer_font.render("Hidden Layer", True, BLACK)
        output_layer_text = layer_font.render("Output Layer", True, BLACK)
        
        self.screen.blit(input_layer_text, (input_x - input_layer_text.get_width()//2, 40))
        self.screen.blit(hidden_layer_text, (hidden_x - hidden_layer_text.get_width()//2, 40))
        self.screen.blit(output_layer_text, (output_x - output_layer_text.get_width()//2, 40))
        
        # Store neuron positions for synapse drawing
        input_positions = []
        hidden_positions = []
        output_positions = []
        
        # Draw synapses first (so they're behind neurons)
        self.draw_synapses(input_x, input_spacing, hidden_x, hidden_spacing, 
                          output_x, output_spacing)
        
        # Draw input neurons
        for i in range(input_size):
            pos_y = (i + 1) * input_spacing + 20  # Add offset for layer labels
            pos = (input_x, pos_y)
            input_positions.append(pos)
            
            # Get activation value from neuron membrane potential
            value = self.brain.input_neurons[i].membrane_potential
            # Normalize value for visualization
            norm_value = min(1.0, max(0, (value + 65.0) / 66.0))
            
            # Get additional neuron info for display
            percepts = self.get_state()
            percept_value = percepts[i] if i < len(percepts) else "N/A"
            
            # Draw neuron with extended info
            self.draw_neuron(pos, value, norm_value, f"In {i}", i < 4, 
                            extra_info=[f"Percept: {percept_value}", f"MP: {value:.2f}mV"])
        
        # Draw hidden neurons
        for i in range(hidden_size):
            pos_y = (i + 1) * hidden_spacing + 20  # Add offset for layer labels
            pos = (hidden_x, pos_y)
            hidden_positions.append(pos)
            
            # Get activation value
            value = self.brain.hidden_neurons[i].membrane_potential
            # Normalize value for visualization
            norm_value = min(1.0, max(0, (value + 65.0) / 66.0))
            
            # Draw neuron with membrane potential info
            self.draw_neuron(pos, value, norm_value, f"H{i}", False,
                           extra_info=[f"MP: {value:.2f}mV"])
        
        # Draw output neurons
        for i in range(output_size):
            pos_y = (i + 1) * output_spacing + 20  # Add offset for layer labels
            pos = (output_x, pos_y)
            output_positions.append(pos)
            
            # Get activation value
            value = self.brain.output_neurons[i].membrane_potential
            # Normalize value for visualization
            norm_value = min(1.0, max(0, (value + 65.0) / 66.0))
            
            # Check if this was the last selected action
            highlight = (self.brain.last_action == i)
            
            # Direction labels
            direction_labels = ["Up", "Down", "Left", "Right"]
            label = direction_labels[i] if i < len(direction_labels) else f"Out {i}"
            
            # Draw neuron with action indication
            action_info = "ACTIVE" if highlight else ""
            self.draw_neuron(pos, value, norm_value, label, highlight,
                            extra_info=[f"MP: {value:.2f}mV", action_info])
            
        # Draw legend
        self.draw_neural_network_legend(self.width + 10, self.nn_height - 90)
    
    def draw_neural_network_legend(self, x, y):
        """Draw a legend for the neural network visualization"""
        legend_font = pygame.font.SysFont("Arial", 12)
        
        # Legend box
        legend_rect = pygame.Rect(x, y, self.nn_width - 20, 80)
        pygame.draw.rect(self.screen, (245, 245, 245), legend_rect)
        pygame.draw.rect(self.screen, BLACK, legend_rect, 1)
        
        # Legend title
        title = legend_font.render("Legend:", True, BLACK)
        self.screen.blit(title, (x + 5, y + 5))
        
        # Neuron activity legend
        pygame.draw.circle(self.screen, (0, 0, 255), (x + 15, y + 25), 7)  # Inactive
        pygame.draw.circle(self.screen, (255, 0, 0), (x + 15, y + 45), 7)  # Active
        inactive_text = legend_font.render("Inactive neuron (-70mV)", True, BLACK)
        active_text = legend_font.render("Active neuron (-55mV+)", True, BLACK)
        self.screen.blit(inactive_text, (x + 30, y + 20))
        self.screen.blit(active_text, (x + 30, y + 40))
        
        # Synapse legend
        pygame.draw.line(self.screen, (0, 200, 0), (x + 200, y + 25), (x + 230, y + 25), 3)
        pygame.draw.line(self.screen, (200, 0, 0), (x + 200, y + 45), (x + 230, y + 45), 3)
        positive_text = legend_font.render("Excitatory synapse (positive weight)", True, BLACK)
        negative_text = legend_font.render("Inhibitory synapse (negative weight)", True, BLACK)
        self.screen.blit(positive_text, (x + 240, y + 20))
        self.screen.blit(negative_text, (x + 240, y + 40))
        
        # Active synapse
        pygame.draw.line(self.screen, YELLOW, (x + 200, y + 65), (x + 230, y + 65), 4)
        active_synapse_text = legend_font.render("Recently fired synapse", True, BLACK)
        self.screen.blit(active_synapse_text, (x + 240, y + 60))
    
    def draw_neuron(self, pos, value, norm_value, label=None, highlight=False, extra_info=None):
        """Draw a neuron circle with color based on activation"""
        x, y = pos
        radius = 12  # Slightly larger for better visibility
        
        # Convert tensor to scalar if needed
        if isinstance(value, torch.Tensor):
            try:
                value = value.item()
            except:
                value = value.mean().item() if value.numel() > 0 else 0.0
        
        # Generate color gradient: blue (0.0) to red (1.0)
        color = (
            int(255 * norm_value),          # Red increases with value
            50,                             # Low green for contrast
            int(255 * (1.0 - norm_value))   # Blue decreases with value
        )
        
        # Draw neuron body
        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        # Draw thicker border for active neurons or highlighted neurons
        border_width = 3 if highlight else 1
        border_color = ORANGE if highlight else BLACK
        pygame.draw.circle(self.screen, border_color, (int(x), int(y)), radius, border_width)
        
        # Add label if provided
        if label:
            label_font = pygame.font.SysFont("Arial", 11, bold=True)
            text = label_font.render(label, True, BLACK)
            self.screen.blit(text, (x - text.get_width() // 2, y - radius - 20))
        
        # Add extra info if provided
        if extra_info:
            info_font = pygame.font.SysFont("Arial", 10)
            for i, info in enumerate(extra_info):
                if info:  # Only display non-empty strings
                    text = info_font.render(info, True, BLACK)
                    # Create a small background for text
                    text_bg = pygame.Rect(
                        x - text.get_width() // 2 - 2,
                        y + radius + 2 + i * 15,
                        text.get_width() + 4,
                        text.get_height()
                    )
                    pygame.draw.rect(self.screen, (240, 240, 240), text_bg)
                    pygame.draw.rect(self.screen, (200, 200, 200), text_bg, 1)
                    # Display text
                    self.screen.blit(text, (x - text.get_width() // 2, y + radius + 2 + i * 15))

        # Draw threshold indicator line inside neuron
        threshold = self.brain.neuron_kwargs['threshold'] if self.brain else -55
        rest = self.brain.neuron_kwargs['rest_potential'] if self.brain else -70
        total_range = threshold - rest
        # Show threshold as a small mark inside the neuron
        if total_range > 0:
            threshold_y = y - int((threshold - rest) / total_range * radius * 0.8)
            pygame.draw.line(self.screen, (50, 50, 50), 
                            (x - radius + 2, threshold_y), 
                            (x + radius - 2, threshold_y), 1)
    
    def draw_synapses(self, input_x, input_spacing, hidden_x, hidden_spacing, 
                     output_x, output_spacing):
        """Draw synapses (connections) between neurons"""
        if not self.brain or not self.brain.synapses:
            return
            
        # Draw each synapse
        for key, synapse in self.brain.synapses.items():
            source_layer, source_idx, target_layer, target_idx = key
            weight = synapse.weight.item() if hasattr(synapse.weight, 'item') else float(synapse.weight)
            
            # Determine source and target positions
            source_pos = None
            target_pos = None
            
            if source_layer == 'input':
                source_pos = (input_x, (source_idx + 1) * input_spacing + 20)  # +20 for layer label offset
            elif source_layer == 'hidden':
                source_pos = (hidden_x, (source_idx + 1) * hidden_spacing + 20)
                
            if target_layer == 'hidden':
                target_pos = (hidden_x, (target_idx + 1) * hidden_spacing + 20)
            elif target_layer == 'output':
                target_pos = (output_x, (target_idx + 1) * output_spacing + 20)
                
            if source_pos and target_pos:
                # Normalize weight based on configurable divisor to scale mV values
                norm_weight = abs(weight) / SYNAPSE_VISUAL_WEIGHT_DIVISOR
                norm_weight = max(0.1, min(1.0, norm_weight))  # Ensure minimum visibility
                
                # Determine line width using configurable scale factor
                width = max(1, int(norm_weight * SYNAPSE_VISUAL_WIDTH_SCALE))
                
                # Use red for negative weights and green for positive weights
                if weight >= 0:
                    color = (0, int(200 * norm_weight), 0)
                else:
                    color = (int(200 * norm_weight), 0, 0)
                    
                is_firing = key in self.firing_synapses
                if is_firing:
                    # Highlight firing synapses with yellow and slightly increase width
                    color = YELLOW
                    width += 1
                
                # Draw synapse line with slight curve for better visualization
                control_point = (
                    (source_pos[0] + target_pos[0]) / 2,
                    (source_pos[1] + target_pos[1]) / 2 + (10 if random.random() > 0.5 else -10)
                )
                
                # Draw curved line approximation using multiple short lines
                points = []
                steps = 20
                for i in range(steps + 1):
                    t = i / steps
                    # Quadratic Bezier curve
                    x = (1-t)**2 * source_pos[0] + 2*(1-t)*t * control_point[0] + t**2 * target_pos[0]
                    y = (1-t)**2 * source_pos[1] + 2*(1-t)*t * control_point[1] + t**2 * target_pos[1]
                    points.append((x, y))
                
                # Draw segmented line for curve effect
                for i in range(len(points) - 1):
                    pygame.draw.line(self.screen, color, points[i], points[i+1], width)
                
                # Draw weight label at the midpoint
                mid_point = points[len(points)//2]
                weight_font = pygame.font.SysFont("Arial", 10)
                weight_text = weight_font.render(f"{weight:.2f}", True, BLACK)
                
                # Create background for weight text
                text_bg = pygame.Rect(
                    mid_point[0] - weight_text.get_width()//2 - 2,
                    mid_point[1] - weight_text.get_height()//2 - 1,
                    weight_text.get_width() + 4,
                    weight_text.get_height() + 2
                )
                pygame.draw.rect(self.screen, WHITE, text_bg)
                pygame.draw.rect(self.screen, (200, 200, 200), text_bg, 1)
                
                # Draw weight text
                self.screen.blit(weight_text, (
                    mid_point[0] - weight_text.get_width()//2,
                    mid_point[1] - weight_text.get_height()//2
                ))
                
                # Draw small arrow at the target end
                if len(points) > 2:
                    end_segment = (points[-2], points[-1])
                    self.draw_arrow_head(end_segment[1], end_segment[0], color, width)

    def draw_arrow_head(self, pos, prev_pos, color, width):
        """Draw a small arrow head at the end of a synapse"""
        # Calculate direction vector
        dx = pos[0] - prev_pos[0]
        dy = pos[1] - prev_pos[1]
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.0001:  # Avoid division by zero
            return
            
        dx, dy = dx/length, dy/length
        
        # Calculate perpendicular vector
        px, py = -dy, dx
        
        # Calculate arrow points
        arrow_size = width + 4
        p1 = (pos[0] - dx*arrow_size - px*arrow_size/2, 
              pos[1] - dy*arrow_size - py*arrow_size/2)
        p2 = (pos[0] - dx*arrow_size + px*arrow_size/2, 
              pos[1] - dy*arrow_size + py*arrow_size/2)
              
        # Draw arrow head
        pygame.draw.polygon(self.screen, color, [pos, p1, p2])
    
    def register_synapse_firing(self, source_layer, source_idx, target_layer, target_idx):
        """Record a synapse that has just fired (for visualization)"""
        key = (source_layer, source_idx, target_layer, target_idx)
        self.firing_synapses.append(key)
