import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
from continues import AdaptiveExponentialLIFNeuron, RewardModulatedSTDPSynapse, NeuralModule, ContinuouslyAdaptingNeuralSystem
from maze_environment import MazeEnvironment  # Import the updated MazeEnvironment

class PerceptionModule(NeuralModule):
    """Neural module for processing sensory input from the maze"""
    def __init__(self, input_size=8, hidden_size=12):
        super(PerceptionModule, self).__init__(
            input_size=input_size,  # 8 surrounding squares
            hidden_size=hidden_size,
            output_size=12,  # Features to feed to the movement module
            module_name="perception"
        )
        self.accepts_sensory = True  # Flag that this module accepts direct sensory input


class MovementModule(NeuralModule):
    """Neural module for controlling movement in the maze"""
    def __init__(self, input_size=12, hidden_size=16):
        super(MovementModule, self).__init__(
            input_size=input_size,  # Features from perception module
            hidden_size=hidden_size,
            output_size=4,  # 4 movement directions
            module_name="movement"
        )
        self.is_motor = True  # Flag that this module produces motor outputs


class MazeBrain:
    """Manages the neural modules for maze navigation"""
    def __init__(self):
        # Create the core neural system
        self.system = ContinuouslyAdaptingNeuralSystem()
        
        # Create modules
        self.perception = PerceptionModule()
        self.movement = MovementModule()
        
        # Add modules to system
        self.system.add_module("perception", self.perception)
        self.system.add_module("movement", self.movement)
        
        # Start the system
        self.system.start()
        
        # For action selection
        self.last_action = None
        self.action_cooldown = 0
        
    def process_observation(self, observation):
        """Process observation and return an action"""
        # Convert observation to tensors if necessary
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # Send to perception module
        self.system.process_sensory_input(observation, "perception")
        
        # Wait for action to be selected
        action = self._get_next_action()
        return action
        
    def _get_next_action(self):
        """Get next action from the movement module"""
        # Wait a bit for processing to happen
        time.sleep(0.05)
        
        # Decrease cooldown
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return self.last_action if self.last_action is not None else 0
        
        # Check action queue
        try:
            module_name, output_spikes = self.system.action_queue.get_nowait()
            if module_name == "movement":
                # Convert output spikes to action
                if isinstance(output_spikes, list) and len(output_spikes) == 4:
                    # Choose the action with highest activation
                    if all(s == 0 for s in output_spikes):
                        # If all outputs are 0, use random action
                        action = random.randint(0, 3)
                    else:
                        action = output_spikes.index(max(output_spikes))
                    
                    # Set cooldown to avoid rapid action changes
                    self.action_cooldown = 2
                    self.last_action = action
                    return action
        except Exception as e:
            # If no action is available, use previous action or default
            pass
            
        # Default action or previous action
        return self.last_action if self.last_action is not None else 0
    
    def provide_reward(self, reward):
        """Provide reward signal to both modules"""
        if reward != 0:
            # Convert reward to float to ensure consistent type
            reward_float = float(reward)
            
            # Apply reward directly to the modules
            for module_name in ["perception", "movement"]:
                if module_name in self.system.modules_dict:
                    module = self.system.modules_dict[module_name]
                    # Apply reward to the module synapses
                    for synapse in module.synapses.values():
                        synapse.apply_reward(reward_float)
            
    def stop(self):
        """Stop the neural system"""
        self.system.stop()


def train_maze_navigation():
    """Train the brain to navigate the maze"""
    # Create environment and brain
    env = MazeEnvironment(width=15, height=15, wall_density=0.2, cell_size=30)  # Updated to use pygame visualization
    brain = MazeBrain()
    
    # Training parameters
    episodes = 50
    render_interval = 5  # Render every N episodes
    
    try:
        for episode in range(episodes):
            # Reset environment
            observation = env.reset()
            total_reward = 0
            done = False
            
            print(f"Starting Episode {episode+1}/{episodes}")
            
            # Run episode
            while not done:
                # Render occasionally
                if episode % render_interval == 0:
                    if not env.render():
                        return
                
                # Get action from brain
                action = brain.process_observation(observation)
                
                # Take action
                new_observation, reward, done = env.step(action)
                total_reward += reward
                
                # Provide reward to brain
                brain.provide_reward(reward)
                
                # Update observation
                observation = new_observation
            
            print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
            
            # Always render the final state
            if episode % render_interval == 0:
                if not env.render():
                    return
                plt.savefig(f"maze_episode_{episode+1}.png")
    
    finally:
        # Stop the brain system
        brain.stop()
        env.close_pygame()  # Ensure pygame is properly closed


if __name__ == "__main__":
    train_maze_navigation()
