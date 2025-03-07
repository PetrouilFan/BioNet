import torch
import torch.nn as nn
import numpy as np
import time
import threading
import queue
import matplotlib.pyplot as plt
from collections import deque
import random

class AdaptiveExponentialLIFNeuron(nn.Module):
    """
    Adaptive Exponential Leaky Integrate-and-Fire Neuron with homeostatic plasticity
    Mimics biological neurons with adaptation mechanisms
    """
    def __init__(self, threshold=1.0, rest_potential=-65.0, reset_potential=-70.0, 
                 leak_constant=0.9, adaptation_constant=0.2, refractory_period=5,
                 excitatory=True):
        super(AdaptiveExponentialLIFNeuron, self).__init__()
        self.threshold = threshold
        self.rest_potential = rest_potential
        self.reset_potential = reset_potential
        self.leak_constant = leak_constant
        self.adaptation_constant = adaptation_constant
        self.refractory_period = refractory_period
        self.excitatory = excitatory  # Determines if neuron is excitatory or inhibitory
        
        # Homeostatic plasticity parameters
        self.target_firing_rate = 0.01  # Target activity level
        self.homeostatic_time_constant = 1000  # Time window for rate estimation
        self.homeostatic_learning_rate = 0.001
        
        # Initialize state
        self.reset()
        
    def reset(self):
        self.membrane_potential = self.rest_potential
        self.adaptation_current = 0.0
        self.refractory_countdown = 0
        self.firing_history = deque(maxlen=self.homeostatic_time_constant)
        self.spike_times = []
        self.potential_history = []
        self.last_spike_time = -1000  # Large negative value for initial state
        
    def forward(self, input_current, time_step):
        """
        Process input current at current time step
        Returns spike (0 or 1) and current membrane potential
        """
        # Ensure input_current is a tensor
        if not torch.is_tensor(input_current):
            input_current = torch.tensor(input_current, dtype=torch.float32)
        # Update membrane potential if not in refractory period
        if self.refractory_countdown <= 0:
            # Leak towards rest potential
            self.membrane_potential = self.leak_constant * (self.membrane_potential - self.rest_potential) + self.rest_potential
            
            # Add input current
            # Ensure membrane_potential tensor has the same shape as input_current
            if not torch.is_tensor(self.membrane_potential) or self.membrane_potential.dim() == 0:
                self.membrane_potential = torch.zeros_like(input_current)
            self.membrane_potential = self.membrane_potential + input_current
            
            # Apply adaptation current (slows down repeated firing)
            self.membrane_potential -= self.adaptation_current
            
        else:
            self.refractory_countdown -= 1
        
        # Check if membrane potential exceeds threshold
        spike = 0.0
        # Handle both scalar and tensor cases properly
        if torch.is_tensor(self.membrane_potential) and self.membrane_potential.numel() > 1:
            # If it's a multi-element tensor, we need to check each element
            if (self.membrane_potential >= self.threshold).any():
                spike = 1.0
                self.membrane_potential = self.reset_potential
                self.adaptation_current += self.adaptation_constant
                self.refractory_countdown = self.refractory_period
                self.last_spike_time = time_step
                self.spike_times.append(time_step)
        else:
            # Scalar or single-element tensor case
            try:
                if self.membrane_potential.item() >= self.threshold:
                    spike = 1.0
                    self.membrane_potential = self.reset_potential
                    self.adaptation_current += self.adaptation_constant
                    self.refractory_countdown = self.refractory_period
                    self.last_spike_time = time_step
                    self.spike_times.append(time_step)
            except:
                # Fallback for any other case
                if torch.any(self.membrane_potential >= self.threshold):
                    spike = 1.0
                    self.membrane_potential = self.reset_potential
                    self.adaptation_current += self.adaptation_constant
                    self.refractory_countdown = self.refractory_period
                    self.last_spike_time = time_step
                    self.spike_times.append(time_step)
        
        # Decay adaptation current
        self.adaptation_current *= 0.95
        
        # Record for visualization and analysis - ensure consistent scalar data types for plotting
        # Convert tensors to simple scalar values before storing
        if torch.is_tensor(self.membrane_potential):
            # If it's a multi-element tensor, store just the first element or mean
            if self.membrane_potential.numel() > 1:
                self.potential_history.append(float(self.membrane_potential.mean()))
            else:
                self.potential_history.append(float(self.membrane_potential.item()))
        else:
            self.potential_history.append(float(self.membrane_potential))
            
        self.firing_history.append(float(spike) if torch.is_tensor(spike) else spike)
        
        # Homeostatic plasticity - adjust threshold based on recent activity
        self._adjust_threshold()
        
        return spike, self.membrane_potential
    
    def _adjust_threshold(self):
        """Implement homeostatic plasticity by adjusting threshold"""
        if len(self.firing_history) >= self.homeostatic_time_constant:
            # Calculate recent firing rate
            recent_rate = sum(self.firing_history) / len(self.firing_history)
            
            # Adjust threshold based on difference from target rate
            rate_diff = recent_rate - self.target_firing_rate
            self.threshold += self.homeostatic_learning_rate * rate_diff
            
            # Ensure threshold stays in reasonable range
            self.threshold = max(0.1, min(self.threshold, 5.0))


class RewardModulatedSTDPSynapse(nn.Module):
    """
    Reward-Modulated Spike-Timing-Dependent Plasticity Synapse
    Integrates reinforcement learning with STDP
    """
    def __init__(self, initial_weight=0.5, learning_rate=0.01, 
                 a_plus=0.2, a_minus=0.21, tau_plus=20.0, tau_minus=20.0,
                 max_weight=1.0, min_weight=0.0, structural_plasticity=True,
                 synapse_id=None):
        super(RewardModulatedSTDPSynapse, self).__init__()
        # Main weight parameter
        self.weight = nn.Parameter(torch.tensor(initial_weight))
        self.learning_rate = learning_rate
        # Increase base learning rate for faster learning
        self.base_learning_rate = learning_rate * 5.0  # Amplify learning rate
        self.enabled = True  # Can be disabled during pruning
        self.synapse_id = synapse_id
        
        # STDP parameters
        self.a_plus = a_plus      # Magnitude of weight potentiation
        self.a_minus = a_minus    # Magnitude of weight depression
        self.tau_plus = tau_plus  # Time constant for potentiation
        self.tau_minus = tau_minus  # Time constant for depression
        
        # Weight constraints
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        # Traces for pre and post synaptic activities
        self.pre_trace = 0.0
        self.post_trace = 0.0
        
        # Eligibility trace for reward-modulated learning
        self.eligibility_trace = 0.0
        self.eligibility_decay = 0.95  # Reduced from 0.95 to 0.90 for slower decay
        
        # Debug information
        self.last_weight_change = 0.0
        self.last_eligibility_value = 0.0
        
        # Structural plasticity parameters
        self.structural_plasticity = structural_plasticity
        self.creation_threshold = 0.2  # Threshold for synaptogenesis
        self.elimination_threshold = 0.01  # Threshold for synaptic pruning
        self.last_usage = 0  # Time step of last significant use
        
        # History for analysis
        self.weight_history = [initial_weight]
        self.eligibility_history = [0.0]
        
    def forward(self, pre_spike, time_step):
        """Propagate spike through the synapse"""
        if not self.enabled:
            return 0.0
            
        # Ensure pre_spike is a tensor
        if not isinstance(pre_spike, torch.Tensor):
            pre_spike = torch.tensor(pre_spike)
        # (Optional) Ensure self.weight is a tensor
        if not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight)
        # Update the last usage time if there's significant activity
        if (pre_spike > 0).any() and (self.weight > 0.1).all():
            self.last_usage = time_step
            
        return pre_spike * self.weight
    
    def update_traces(self, pre_spike, post_spike, time_step):
        """Update STDP traces based on spike timing"""
        if not self.enabled:
            return
            
        # Ensure pre_spike and post_spike are tensors
        if not isinstance(pre_spike, torch.Tensor):
            pre_spike = torch.tensor(pre_spike, dtype=torch.float32)
        if not isinstance(post_spike, torch.Tensor):
            post_spike = torch.tensor(post_spike, dtype=torch.float32)
            
        # Update exponential traces (time-sensitive memory)
        self.pre_trace = self.pre_trace * torch.exp(torch.tensor(-1.0/self.tau_plus)) + pre_spike
        self.post_trace = self.post_trace * torch.exp(torch.tensor(-1.0/self.tau_minus)) + post_spike
        
        # Calculate STDP update based on relative timing
        stdp_update = torch.tensor(0.0)
        
        # Check if post-synaptic neuron fired (handle both scalar and tensor cases)
        if isinstance(post_spike, torch.Tensor) and post_spike.numel() > 1:
            if torch.any(post_spike > 0):  # Fix: Changed post_spike.any() > 0 to torch.any(post_spike > 0)
                stdp_update += self.a_plus * self.pre_trace
        elif post_spike > 0:  # Scalar or single-element tensor case
            stdp_update += self.a_plus * self.pre_trace
        
        # Check if pre-synaptic neuron fired (handle both scalar and tensor cases)
        if isinstance(pre_spike, torch.Tensor) and pre_spike.numel() > 1:
            if torch.any(pre_spike > 0):  # Fix: Changed pre_spike.any() > 0 to torch.any(pre_spike > 0)
                stdp_update -= self.a_minus * self.post_trace
        elif pre_spike > 0:  # Scalar or single-element tensor case
            stdp_update -= self.a_minus * self.post_trace
        
        # Add noise to eligibility trace to break symmetry and encourage exploration
        noise = torch.randn(1).item() * 0.01  # Small random noise
        
        # Update eligibility trace (candidate for weight change)
        self.eligibility_trace = self.eligibility_trace * self.eligibility_decay + stdp_update + noise
        
        # Store for debugging
        self.last_eligibility_value = float(self.eligibility_trace) if torch.is_tensor(self.eligibility_trace) else self.eligibility_trace
        
        # Record history
        self.eligibility_history.append(self.last_eligibility_value)
    
    def apply_reward(self, reward_signal):
        """Apply reward-modulated weight change using eligibility trace"""
        if not self.enabled:
            return
            
        # Modified: Force a minimum eligibility value when trace is too small
        # This ensures rewards always cause meaningful weight changes
        if abs(self.eligibility_trace) < 0.1:
            # Use the sign of the existing trace if available, otherwise random sign
            sign = 1.0 if self.eligibility_trace > 0 else (-1.0 if self.eligibility_trace < 0 else (1.0 if random.random() > 0.5 else -1.0))
            self.eligibility_trace = sign * 0.1  # Force substantial eligibility for effective learning
            
        # Convert to float if tensor to avoid gradient tracking issues
        if torch.is_tensor(self.eligibility_trace):
            eligibility_value = self.eligibility_trace.item()
        else:
            eligibility_value = self.eligibility_trace
            
        # Modify weight based on reward and eligibility (use amplified learning rate)
        weight_change = self.base_learning_rate * reward_signal * eligibility_value
        
        # Store for debugging
        self.last_weight_change = weight_change
        
        # Apply weight change to tensor data
        if self.weight.is_leaf:  # Check if weight is a leaf variable we can modify
            self.weight.data += weight_change
            
            # Constrain weight to valid range
            self.weight.data.clamp_(self.min_weight, self.max_weight)
            
            # Add current weight to history for tracking
            if len(self.weight_history) > 100:  # Limit history size
                self.weight_history = self.weight_history[-50:]
            self.weight_history.append(self.weight.item())
            
            # Debug info if significant change
            if abs(weight_change) > 0.001:
                print(f"Weight changed by {weight_change:.4f} to {self.weight.item():.4f} (elig:{eligibility_value:.4f}, reward:{reward_signal:.2f})")
    
    def check_structural_plasticity(self, network_activity, time_step, current_density):
        """
        Check if this synapse should be pruned or strengthened based on usage
        Returns: (prune, strengthen)
        """
        if not self.structural_plasticity:
            return False, False
            
        # Calculate time since last usage
        time_since_use = time_step - self.last_usage
        
        # Check for pruning - remove synapses that are weak and unused
        prune = (self.weight < self.elimination_threshold and 
                 time_since_use > 1000 and  # Unused for long time
                 current_density > 0.2)  # Only prune if we have enough connections
        
        # Check for strengthening - boost synapses with accumulated eligibility
        strengthen = abs(self.eligibility_trace) > self.creation_threshold
        
        return prune, strengthen


class NeuralModule(nn.Module):
    """
    A modular component with input and output neurons and internal connectivity
    Designed for dynamic rewiring and integration with other modules
    """
    def __init__(self, input_size, hidden_size, output_size, module_name="generic"):
        super(NeuralModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.module_name = module_name
        
        # Create neurons for each layer (80% excitatory, 20% inhibitory)
        self.input_neurons = nn.ModuleList([
            AdaptiveExponentialLIFNeuron(excitatory=(i < int(0.8 * input_size))) 
            for i in range(input_size)
        ])
        
        self.hidden_neurons = nn.ModuleList([
            AdaptiveExponentialLIFNeuron(excitatory=(i < int(0.8 * hidden_size))) 
            for i in range(hidden_size)
        ])
        
        self.output_neurons = nn.ModuleList([
            AdaptiveExponentialLIFNeuron(excitatory=True)  # Output neurons are all excitatory
            for _ in range(output_size)
        ])
        
        # Create modifiable synapse dictionary for dynamic connectivity
        # Format: (source_layer, source_idx, target_layer, target_idx) -> synapse
        self.synapses = {}
        self.next_synapse_id = 0
        
        # Initialize with sparse random connectivity
        self._initialize_connectivity()
        
        # Module state
        self.activity_level = 0.0
        self.reward_buffer = deque(maxlen=100)  # Store recent rewards
        self.is_active = True  # Whether this module is currently engaged
        self.time_step = 0
        
        # For tracking module performance
        self.performance_history = []
        
    def _initialize_connectivity(self):
        """Create initial sparse random connectivity"""
        # Input to hidden layer (30% connectivity)
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                if random.random() < 0.3:  # Sparse connectivity
                    self._create_synapse('input', i, 'hidden', h)
        
        # Hidden to hidden layer (recurrent, 20% connectivity)
        for h1 in range(self.hidden_size):
            for h2 in range(self.hidden_size):
                if h1 != h2 and random.random() < 0.2:
                    self._create_synapse('hidden', h1, 'hidden', h2)
        
        # Hidden to output layer (50% connectivity)
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                if random.random() < 0.5:
                    self._create_synapse('hidden', h, 'output', o)
    
    def _create_synapse(self, source_layer, source_idx, target_layer, target_idx, 
                       initial_weight=None):
        """Create a new synapse between neurons"""
        # Generate random weight if not provided
        if initial_weight is None:
            initial_weight = random.uniform(0.1, 0.5)
        
        # Create synapse with unique ID
        synapse = RewardModulatedSTDPSynapse(
            initial_weight=initial_weight,
            synapse_id=self.next_synapse_id
        )
        self.next_synapse_id += 1
        
        # Store in synapse dictionary
        key = (source_layer, source_idx, target_layer, target_idx)
        self.synapses[key] = synapse
        
        return synapse
    
    def _get_neuron(self, layer, idx):
        """Helper to get neuron by layer and index"""
        if layer == 'input':
            return self.input_neurons[idx]
        elif layer == 'hidden':
            return self.hidden_neurons[idx]
        elif layer == 'output':
            return self.output_neurons[idx]
        raise ValueError(f"Unknown layer: {layer}")
        
    def forward(self, input_signals, reward_signal=0.0):
        """
        Process input signals and update network state
        Returns output spikes and internal state
        """
        self.time_step += 1
        
        # Convert inputs to spikes for input layer
        input_spikes = []
        for i, signal in enumerate(input_signals):
            # Input neurons just pass through the signal as spikes
            if isinstance(signal, torch.Tensor) and signal.requires_grad:
                signal = signal.detach()
            if isinstance(signal, (list, np.ndarray, torch.Tensor)) and len(np.atleast_1d(signal)) > 1:
                # Handle array case
                should_spike = np.zeros_like(signal, dtype=bool)
                for i, s in enumerate(signal):
                    if s > 0 and random.random() < s:
                        should_spike[i] = True
                input_spikes.append(should_spike)
            else:
                # Handle scalar case
                if signal > 0 and random.random() < signal:  # Probabilistic spiking
                    spike = 1.0
                else:
                    spike = 0.0
                input_spikes.append(spike)
        
        # Initialize layer activities
        hidden_inputs = [0.0] * self.hidden_size
        output_inputs = [0.0] * self.output_size
        
        # Process input to hidden connections
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                key = ('input', i, 'hidden', h)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    # Forward pass through synapse
                    current = synapse(input_spikes[i], self.time_step)
                    hidden_inputs[h] += current
        
        # Process hidden layer activity
        hidden_spikes = []
        for h in range(self.hidden_size):
            spike, _ = self.hidden_neurons[h](hidden_inputs[h], self.time_step)
            hidden_spikes.append(spike)
        
        # Process hidden to hidden recurrent connections
        for h1 in range(self.hidden_size):
            if hidden_spikes[h1] > 0:  # Only if this neuron spiked
                for h2 in range(self.hidden_size):
                    key = ('hidden', h1, 'hidden', h2)
                    if key in self.synapses:
                        # Add delayed recurrent input (will affect next time step)
                        synapse = self.synapses[key]
                        self.hidden_neurons[h2].membrane_potential += synapse(hidden_spikes[h1], self.time_step)
        
        # Process hidden to output connections
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                key = ('hidden', h, 'output', o)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    current = synapse(hidden_spikes[h], self.time_step)
                    output_inputs[o] += current
        
        # Process output layer activity
        output_spikes = []
        for o in range(self.output_size):
            spike, _ = self.output_neurons[o](output_inputs[o], self.time_step)
            output_spikes.append(spike)
            
        # Store overall network activity level (for homeostasis)
        self.activity_level = 0.6 * self.activity_level + 0.4 * (
            sum(input_spikes) / len(input_spikes) +
            sum(hidden_spikes) / len(hidden_spikes) +
            sum(output_spikes) / len(output_spikes)
        ) / 3.0
        
        # Update all synapse traces
        self._update_synapse_traces(input_spikes, hidden_spikes, output_spikes)
        
        # Apply reward modulation if there's a reward signal
        if reward_signal != 0:
            self.reward_buffer.append(reward_signal)
            self._apply_reward(reward_signal)
            
        # Perform structural plasticity (synaptogenesis & pruning)
        if self.time_step % 1000 == 0:  # Check every 1000 time steps
            self._structural_plasticity()
            
        return output_spikes
    
    def _update_synapse_traces(self, input_spikes, hidden_spikes, output_spikes):
        """Update all synapse traces based on pre and post activity"""
        # Update input to hidden synapses
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                key = ('input', i, 'hidden', h)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    synapse.update_traces(input_spikes[i], hidden_spikes[h], self.time_step)
        
        # Update hidden to hidden synapses
        for h1 in range(self.hidden_size):
            for h2 in range(self.hidden_size):
                key = ('hidden', h1, 'hidden', h2)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    synapse.update_traces(hidden_spikes[h1], hidden_spikes[h2], self.time_step)
        
        # Update hidden to output synapses
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                key = ('hidden', h, 'output', o)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    synapse.update_traces(hidden_spikes[h], output_spikes[o], self.time_step)
    
    def _apply_reward(self, reward_signal):
        """Apply reward to all synapses based on eligibility traces"""
        for synapse in self.synapses.values():
            synapse.apply_reward(reward_signal)
    
    def _structural_plasticity(self):
        """Implement structural plasticity: pruning and synaptogenesis"""
        # Count current connections for density calculation
        total_possible = (self.input_size * self.hidden_size + 
                         self.hidden_size * self.hidden_size + 
                         self.hidden_size * self.output_size)
        current_count = len(self.synapses)
        current_density = current_count / total_possible
        
        # Synaptic pruning - remove weak, unused synapses
        keys_to_remove = []
        for key, synapse in self.synapses.items():
            prune, _ = synapse.check_structural_plasticity(
                self.activity_level, self.time_step, current_density)
            if prune:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.synapses[key]
            
        # Synaptogenesis - create new synapses where needed
        # Prioritize neurons with high activity but few connections
        if current_density < 0.4:  # Keep overall density in check
            # 1. Find underconnected neurons with high activity
            neuron_activity = {}
            neuron_connections = {}
            
            # Analyze hidden neurons
            for h in range(self.hidden_size):
                activity = sum(1 for spike in self.hidden_neurons[h].firing_history if spike > 0)
                neuron_activity[('hidden', h)] = activity
                neuron_connections[('hidden', h)] = 0
            
            # Count existing connections
            for key in self.synapses:
                target = (key[2], key[3])
                if target in neuron_connections:
                    neuron_connections[target] += 1
            
            # Find underconnected but active neurons
            candidates = []
            for neuron, activity in neuron_activity.items():
                if activity > 50 and neuron_connections[neuron] < 5:
                    candidates.append(neuron)
            
            # Create new connections to these neurons
            for target in candidates[:5]:  # Limit to 5 new connections per update
                layer, idx = target
                
                if layer == 'hidden':
                    # Add input to hidden connection
                    source_idx = random.randint(0, self.input_size - 1)
                    self._create_synapse('input', source_idx, 'hidden', idx)
                    
                    # Add recurrent connection
                    source_idx = random.randint(0, self.hidden_size - 1)
                    if source_idx != idx:  # Avoid self-connections
                        self._create_synapse('hidden', source_idx, 'hidden', idx)
    
    def get_performance_stats(self):
        """Get statistics about this module's performance"""
        stats = {
            'activity_level': self.activity_level,
            'synapse_count': len(self.synapses),
            'mean_reward': np.mean(self.reward_buffer) if self.reward_buffer else 0,
            'neuron_count': self.input_size + self.hidden_size + self.output_size
        }
        return stats


class ContinuouslyAdaptingNeuralSystem(nn.Module):
    """
    Main system that coordinates multiple neural modules
    Implements continuous adaptation and lifelong learning
    """
    def __init__(self):
        super(ContinuouslyAdaptingNeuralSystem, self).__init__()
        
        # Store neural modules for different functionalities
        self.modules_dict = nn.ModuleDict()
        
        # Track inter-module connections
        self.cross_module_connections = {}
        
        # For monitoring overall system performance
        self.global_time = 0
        self.performance_history = []
        
        # Queue for sensory input and action output
        self.sensory_queue = queue.Queue()
        self.action_queue = queue.Queue()
        
        # Threads for continuous operation
        self.is_running = False
        self.threads = []
        
    def add_module(self, module_name, neural_module):
        """Add a new functional module to the system"""
        self.modules_dict[module_name] = neural_module
        
        # Establish sparse random connections to existing modules
        for existing_name, existing_module in self.modules_dict.items():
            if existing_name != module_name:
                self._connect_modules(module_name, existing_name)
    
    def _connect_modules(self, module1_name, module2_name, connection_density=0.1):
        """Create cross-module connections for information transfer"""
        module1 = self.modules_dict[module1_name]
        module2 = self.modules_dict[module2_name]
        
        # Bidirectional connections between output of one module and input of the other
        # Module 1 → Module 2
        for o in range(module1.output_size):
            for i in range(module2.input_size):
                if random.random() < connection_density:
                    synapse = RewardModulatedSTDPSynapse()
                    key = (module1_name, 'output', o, module2_name, 'input', i)
                    self.cross_module_connections[key] = synapse
        
        # Module 2 → Module 1
        for o in range(module2.output_size):
            for i in range(module1.input_size):
                if random.random() < connection_density:
                    synapse = RewardModulatedSTDPSynapse()
                    key = (module2_name, 'output', o, module1_name, 'input', i)
                    self.cross_module_connections[key] = synapse
    
    def process_sensory_input(self, sensory_data, source_module=None):
        """Process incoming sensory data and distribute to relevant modules"""
        # Add sensory input to the queue
        self.sensory_queue.put((sensory_data, source_module))
    
    def _continuous_adaptation_loop(self):
        """Main processing loop that runs continuously"""
        while self.is_running:
            self.global_time += 1
            
            # 1. Process any sensory inputs in the queue
            sensory_inputs = {}
            while not self.sensory_queue.empty():
                try:
                    data, target_module = self.sensory_queue.get_nowait()
                    if target_module:
                        sensory_inputs[target_module] = data
                    else:
                        # Distribute to all modules that accept sensory input
                        for name, module in self.modules_dict.items():
                            if hasattr(module, 'accepts_sensory') and module.accepts_sensory:
                                sensory_inputs[name] = data
                except queue.Empty:
                    break
            
            # 2. Update each module
            module_outputs = {}
            for name, module in self.modules_dict.items():
                # Get inputs for this module
                inputs = []
                
                # Add any direct sensory input
                if name in sensory_inputs:
                    inputs.append(sensory_inputs[name])
                
                # Add cross-module inputs from other modules' previous outputs
                for key, synapse in self.cross_module_connections.items():
                    source_module, _, source_idx, target_module, _, target_idx = key
                    if target_module == name:
                        if source_module in module_outputs:
                            source_output = module_outputs[source_module]
                            if source_idx < len(source_output):
                                # Pass input through the cross-module synapse
                                signal = synapse(source_output[source_idx], self.global_time)
                                # Convert signal to scalar if possible
                                if hasattr(signal, 'item'):
                                    signal = signal.item()
                                if len(inputs) <= target_idx:
                                    # Extend inputs list if needed
                                    inputs.extend([0] * (target_idx - len(inputs) + 1))
                                inputs[target_idx] += signal
                
                # Ensure inputs list matches module's input size
                if len(inputs) < module.input_size:
                    inputs.extend([0] * (module.input_size - len(inputs)))
                
                # Forward pass through the module
                output = module(inputs[:module.input_size])
                module_outputs[name] = output
                
                # Send motor outputs to action queue if this is a motor module
                if hasattr(module, 'is_motor') and module.is_motor:
                    self.action_queue.put((name, output))
            
            # 3. Evaluate system performance and adjust cross-module connections
            if self.global_time % 1000 == 0:
                self._evaluate_and_adjust()
                
            # Sleep briefly to avoid consuming all CPU
            time.sleep(0.001)
    
    def _evaluate_and_adjust(self):
        """Evaluate overall system performance and adjust connections accordingly"""
        # Get performance stats from each module
        module_stats = {name: module.get_performance_stats() 
                       for name, module in self.modules_dict.items()}
        
        # Overall system metrics
        system_metrics = {
            'time': self.global_time,
            'module_count': len(self.modules_dict),
            'cross_connections': len(self.cross_module_connections),
            'average_activity': np.mean([stats['activity_level'] for stats in module_stats.values()])
        }
        self.performance_history.append(system_metrics)
        
        # Adjust cross-module connections based on correlation of activity
        # Strengthen connections between modules that are active together
        
        # Track usage patterns for each cross-module connection
        if self.global_time % 5000 == 0:  # Less frequent structural changes
            # 1. Identify and prune unused connections
            keys_to_remove = []
            for key, synapse in self.cross_module_connections.items():
                if synapse.weight < 0.05 and random.random() < 0.3:  # Probabilistic pruning
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cross_module_connections[key]
            
            # 2. Create new connections between active modules
            active_modules = [name for name, stats in module_stats.items() 
                             if stats['activity_level'] > 0.1]
            
            if len(active_modules) >= 2:
                for _ in range(5):  # Add a few new connections
                    module1 = random.choice(active_modules)
                    module2 = random.choice(active_modules)
                    if module1 != module2:
                        self._add_random_connection(module1, module2)
    
    def _add_random_connection(self, module1_name, module2_name):
        """Add a new random connection between two modules"""
        module1 = self.modules_dict[module1_name]
        module2 = self.modules_dict[module2_name]
        
        # Select random neurons to connect
        output_idx = random.randint(0, module1.output_size - 1)
        input_idx = random.randint(0, module2.input_size - 1)
        
        # Create new synapse
        synapse = RewardModulatedSTDPSynapse(initial_weight=random.uniform(0.3, 0.7))
        key = (module1_name, 'output', output_idx, module2_name, 'input', input_idx)
        
        # Only add if connection doesn't already exist
        if key not in self.cross_module_connections:
            self.cross_module_connections[key] = synapse
    
    def start(self):
        """Start the continuous adaptation loop"""
        if not self.is_running:
            self.is_running = True
            
            # Start processing thread
            processing_thread = threading.Thread(
                target=self._continuous_adaptation_loop, 
                daemon=True
            )
            processing_thread.start()
            self.threads.append(processing_thread)
            
            print("Neural system started - continuously adapting")
    
    def stop(self):
        """Stop the continuous adaptation loop"""
        self.is_running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
