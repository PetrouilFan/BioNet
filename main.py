import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading
import queue  # Add missing queue import
from continues import AdaptiveExponentialLIFNeuron, RewardModulatedSTDPSynapse, NeuralModule, ContinuouslyAdaptingNeuralSystem

class PerceptionModule(NeuralModule):
    """Module that processes sensory input"""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(PerceptionModule, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            module_name="perception"
        )
        self.accepts_sensory = True  # Flag to receive external sensory input

class MotorModule(NeuralModule):
    """Module that controls motor outputs"""
    def __init__(self, input_size=5, hidden_size=15, output_size=3):
        super(MotorModule, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            module_name="motor"
        )
        self.is_motor = True  # Flag to designate as motor output module

class AssociationModule(NeuralModule):
    """Module that forms associations between different inputs"""
    def __init__(self, input_size=8, hidden_size=25, output_size=8):
        super(AssociationModule, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            module_name="association"
        )
    
    def forward(self, signal):
        """Override forward method to safely handle tensors with gradients"""
        # Convert tensor with gradients to numpy safely if needed
        if isinstance(signal, torch.Tensor) and signal.requires_grad:
            # Use detach() before any potential numpy conversion
            signal_processed = signal.detach()
            # Pass the detached tensor to parent's implementation
            return super().forward(signal_processed)
        else:
            return super().forward(signal)

class Visualizer:
    """Visualizes neural activity and network structure"""
    def __init__(self, neural_system):
        self.neural_system = neural_system
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.4)
        
        # Initialize plots
        self.module_activity = {name: [] for name in self.neural_system.modules_dict}
        self.connection_strength = []
        self.rewards_history = []
        self.time_points = []
        
        # Create plots
        self._setup_plots()
        self.data_queue = queue.Queue()
        
    def _setup_plots(self):
        """Initialize all plot elements"""
        # Module activity plot
        self.axes[0, 0].set_title('Module Activity Levels')
        self.axes[0, 0].set_xlabel('Time')
        self.axes[0, 0].set_ylabel('Activity')
        self.activity_lines = {}
        
        for name in self.module_activity:
            line, = self.axes[0, 0].plot([], [], label=name)
            self.activity_lines[name] = line
            
        self.axes[0, 0].legend()
        
        # Connectivity plot (network graph visualization)
        self.axes[0, 1].set_title('Cross-Module Connectivity')
        self.axes[0, 1].axis('off')
        
        # Spike raster plot for neurons
        self.axes[1, 0].set_title('Spike Raster Plot')
        self.axes[1, 0].set_xlabel('Time Step')
        self.axes[1, 0].set_ylabel('Neuron Index')
        
        # Weight distribution histogram
        self.axes[1, 1].set_title('Synaptic Weight Distribution')
        self.axes[1, 1].set_xlabel('Weight Value')
        self.axes[1, 1].set_ylabel('Frequency')
        
        # System metrics plot
        self.axes[2, 0].set_title('System Performance Metrics')
        self.axes[2, 0].set_xlabel('Time')
        self.axes[2, 0].set_ylabel('Value')
        
        # Reward history
        self.axes[2, 1].set_title('Reward History')
        self.axes[2, 1].set_xlabel('Time')
        self.axes[2, 1].set_ylabel('Reward')
        
    def update_data(self):
        """Collect latest data from the neural system"""
        self.time_points.append(self.neural_system.global_time)
        
        # Record module activity
        for name, module in self.neural_system.modules_dict.items():
            stats = module.get_performance_stats()
            if name not in self.module_activity:
                self.module_activity[name] = []
            # Ensure the activity value is a scalar
            val = stats['activity_level']
            if isinstance(val, (list, tuple, np.ndarray)):
                val = np.mean(val)
            self.module_activity[name].append(val)
        
        # Record connection strengths
        conn_strengths = [synapse.weight.item() for synapse in self.neural_system.cross_module_connections.values()]
        self.connection_strength.append(np.mean(conn_strengths) if conn_strengths else 0)
        
        # Update rewards history based on module rewards
        avg_reward = np.mean([stats.get('mean_reward', 0) for name, stats in 
                              {n: m.get_performance_stats() for n, m in self.neural_system.modules_dict.items()}.items()])
        self.rewards_history.append(avg_reward)
        self.data_queue.put(True)  # Signal that new data is ready
        
        # Modified: Update performance metrics for visualization with safe averaging
        if not hasattr(self.neural_system, 'performance_history') or not self.neural_system.performance_history:
            self.neural_system.performance_history = []
        activities = []
        for module in self.neural_system.modules_dict.values():
            stat = module.get_performance_stats().get('activity_level', 0)
            if isinstance(stat, (list, tuple, np.ndarray)):
                stat = np.mean(stat)
            activities.append(stat)
        self.neural_system.performance_history.append({
            'time': self.neural_system.global_time,
            'cross_connections': len(self.neural_system.cross_module_connections),
            'average_activity': np.mean(activities)
        })
        
    def update_plots(self):
        """Update visualization with latest data"""
        # Update activity plots
        for name, activity in self.module_activity.items():
            if name in self.activity_lines:
                self.activity_lines[name].set_data(self.time_points[-100:], activity[-100:])
                
        # Auto-scale axes
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
            
        # Update network graph
        self._update_network_graph()
        
        # Update spike raster
        self._update_spike_raster()
        
        # Update weight histogram
        self._update_weight_histogram()
        
        # Update system metrics
        self._plot_system_metrics()
        
        # Update reward history
        self.axes[2, 1].clear()
        self.axes[2, 1].set_title('Reward History')
        self.axes[2, 1].plot(self.time_points[-100:], self.rewards_history[-100:])
        
        self.fig.canvas.draw()
        
    def _update_network_graph(self):
        """Update network graph visualization"""
        self.axes[0, 1].clear()
        self.axes[0, 1].set_title('Cross-Module Connectivity')
        
        # Create a simple representation of modules and connections
        module_positions = {}
        modules = list(self.neural_system.modules_dict.keys())
        
        # Position modules in a circle
        num_modules = len(modules)
        for i, name in enumerate(modules):
            angle = 2 * np.pi * i / num_modules
            x, y = 0.5 * np.cos(angle) + 0.5, 0.5 * np.sin(angle) + 0.5
            module_positions[name] = (x, y)
            
            # Draw module as a circle
            circle = plt.Circle((x, y), 0.1, fill=True, alpha=0.7)
            self.axes[0, 1].add_patch(circle)
            self.axes[0, 1].text(x, y, name, ha='center', va='center')
            
        # Draw connections
        for key, synapse in self.neural_system.cross_module_connections.items():
            source_mod, _, _, target_mod, _, _ = key
            if source_mod in module_positions and target_mod in module_positions:
                x1, y1 = module_positions[source_mod]
                x2, y2 = module_positions[target_mod]
                
                # Line width based on weight
                width = max(0.5, min(3.5, synapse.weight.item() * 5))
                self.axes[0, 1].plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=width)
                
        self.axes[0, 1].set_xlim(0, 1)
        self.axes[0, 1].set_ylim(0, 1)
        self.axes[0, 1].axis('off')
        
    def _update_spike_raster(self):
        """Update spike raster plot"""
        self.axes[1, 0].clear()
        self.axes[1, 0].set_title('Spike Raster Plot')
        
        # Sample neurons to display
        spike_data = []
        labels = []
        
        # Get sample neurons from each module
        neuron_idx = 0
        for name, module in self.neural_system.modules_dict.items():
            # Get a sample of neurons from each layer
            for i in range(min(2, module.input_size)):
                if len(module.input_neurons[i].spike_times) > 0:
                    times = [t for t in module.input_neurons[i].spike_times 
                             if self.neural_system.global_time - 100 <= t <= self.neural_system.global_time]
                    spike_data.append((neuron_idx, times))
                    labels.append(f"{name}-in{i}")
                    neuron_idx += 1
                    
            for i in range(min(2, module.hidden_size)):
                if len(module.hidden_neurons[i].spike_times) > 0:
                    times = [t for t in module.hidden_neurons[i].spike_times 
                             if self.neural_system.global_time - 100 <= t <= self.neural_system.global_time]
                    spike_data.append((neuron_idx, times))
                    labels.append(f"{name}-hid{i}")
                    neuron_idx += 1
            
        # Plot spike raster
        for i, (idx, times) in enumerate(spike_data):
            if times:
                self.axes[1, 0].scatter(times, [i] * len(times), marker='|', s=10, c='black')
        if not spike_data:
            self.axes[1, 0].text(0.5, 0.5, "No spikes recorded", ha='center', va='center')
        
        # Set y-ticks to neuron labels
        self.axes[1, 0].set_yticks(range(len(labels)))
        self.axes[1, 0].set_yticklabels(labels)
        self.axes[1, 0].set_xlabel('Time Step')
        
    def _update_weight_histogram(self):
        """Update weight distribution histogram"""
        self.axes[1, 1].clear()
        # Modified: update title with mean weight value for dynamic feedback
        weights = []
        
        # Get module-internal weights
        for module in self.neural_system.modules_dict.values():
            for synapse in module.synapses.values():
                weights.append(synapse.weight.item())
        
        # Get cross-module weights
        for synapse in self.neural_system.cross_module_connections.values():
            weights.append(synapse.weight.item())
            
        mean_weight = np.mean(weights) if weights else 0
        self.axes[1, 1].set_title(f'Synaptic Weight Distribution (mean: {mean_weight:.2f})')
        
        if weights:
            self.axes[1, 1].hist(weights, bins=20, alpha=0.7)
            self.axes[1, 1].set_xlabel('Weight Value')
            self.axes[1, 1].set_ylabel('Frequency')
        
    def _plot_system_metrics(self):
        """Plot various system metrics"""
        self.axes[2, 0].clear()
        self.axes[2, 0].set_title('System Performance Metrics')
        
        if self.neural_system.performance_history:
            times = [metric['time'] for metric in self.neural_system.performance_history]
            
            # Plot connection count
            connections = [metric['cross_connections'] for metric in self.neural_system.performance_history]
            self.axes[2, 0].plot(times, connections, label='Cross-Connections', marker='o')
            
            # Plot average activity
            activities = [metric['average_activity'] for metric in self.neural_system.performance_history]
            self.axes[2, 0].plot(times, activities, label='Avg Activity', marker='x')
            
            self.axes[2, 0].legend()
            self.axes[2, 0].set_xlabel('Time')
            
    def start_visualization(self, update_interval=1000):
        """Start visualization in separate thread"""
        def update_loop():
            while self.neural_system.is_running:
                self.update_data()
                # Don't update plots here, just queue the data
                time.sleep(update_interval / 1000)  # Convert to seconds
                
        viz_thread = threading.Thread(target=update_loop, daemon=True)
        viz_thread.start()
        return viz_thread

class SimpleEnvironment:
    """Simple environment to test the neural system"""
    def __init__(self, neural_system):
        self.neural_system = neural_system
        self.state = np.zeros(10)  # Environment state
        self.reward = 0
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start environment simulation in separate thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop environment simulation"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _run_simulation(self):
        """Run the main environment simulation loop"""
        import queue  # Local import to ensure availability
        
        while self.is_running:
            # 1. Update environment state with some random changes
            self.state = 0.8 * self.state + 0.2 * np.random.uniform(-1, 1, size=10)
            
            # Convert state to input signals (0 to 1 range)
            input_signals = np.clip((self.state + 1) / 2, 0, 1)
            
            # 2. Send state as sensory input to neural system
            self.neural_system.process_sensory_input(input_signals)
            
            # 3. Get any motor actions from the system
            try:
                module_name, action = self.neural_system.action_queue.get(timeout=0.1)
                
                # 4. Compute reward based on action
                # Simple reward: encourage action values to match a target pattern
                target_pattern = np.array([0.7, 0.3, 0.9])
                action_array = np.array(action)
                
                # Distance-based reward (closer to target = higher reward)
                distance = np.sum(np.abs(target_pattern - action_array))
                self.reward = max(0, 1 - distance/len(target_pattern))
                
                # Send reward signal back to the system
                if module_name in self.neural_system.modules_dict:
                    # Send reward directly to the module that produced the action
                    module = self.neural_system.modules_dict[module_name]
                    signals = np.zeros(module.input_size)
                    signals[0] = self.reward  # Use first input as reward channel
                    self.neural_system.process_sensory_input(signals, module_name)
                    
            except queue.Empty:  # Now queue should be properly recognized
                # No action output yet
                pass
                
            time.sleep(0.05)  # 50ms timestep

def main():
    """Main function to set up and run the neural system"""
    print("Initializing Neural System...")
    
    # Create neural system
    system = ContinuouslyAdaptingNeuralSystem()
    
    # Create modules
    perception = PerceptionModule(input_size=10, hidden_size=20, output_size=5)
    association = AssociationModule(input_size=8, hidden_size=25, output_size=8)
    motor = MotorModule(input_size=5, hidden_size=15, output_size=3)
    
    # Add modules to system
    system.add_module("perception", perception)
    system.add_module("association", association)
    system.add_module("motor", motor)
    
    # Create environment
    env = SimpleEnvironment(system)
    
    # Create visualizer
    visualizer = Visualizer(system)
    
    # Start all components
    print("Starting simulation...")
    system.start()
    env.start()
    
    # Start visualization
    viz_thread = visualizer.start_visualization(update_interval=500)  # Update every 500ms
    
    # Show plots
    plt.ion()  # Interactive mode
    plt.show()
    
    try:
        print("Running... (Press Ctrl+C to stop)")
        while True:
            # Check if there's data to update and update plots from main thread
            try:
                data_ready = visualizer.data_queue.get(block=False)
                if data_ready:
                    visualizer.update_plots()
            except queue.Empty:
                pass
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean shutdown
        env.stop()
        system.stop()
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    main()
