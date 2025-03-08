import torch
import torch.nn as nn
import numpy as np
import time
import threading
import queue
import matplotlib.pyplot as plt
from collections import deque
import random

# GLOBAL CONFIGURATION PARAMETERS
# ==============================

# NEURON PARAMETERS
# -----------------
NEURON_THRESHOLD = -55.0           # mV - membrane potential threshold for firing
NEURON_REST_POTENTIAL = -65.0      # mV - resting membrane potential
NEURON_RESET_POTENTIAL = -70.0     # mV - after-spike reset potential
NEURON_LEAK_CONSTANT = 0.95         # decay factor toward rest potential
NEURON_ADAPTATION_CONSTANT = 0.15   # increment to adaptation current after spike
NEURON_REFRACTORY_PERIOD = 5       # time steps neuron is unresponsive after spike
NEURON_ADAPTATION_DECAY = 0.95     # decay rate for adaptation current
NEURON_TARGET_FIRING_RATE = 0.01   # target activity level for homeostasis
NEURON_HOMEOSTATIC_TIME_CONSTANT = 1000  # time window for rate estimation
NEURON_HOMEOSTATIC_LEARNING_RATE = 0.001 # learning rate for threshold adaptation

# SYNAPSE PARAMETERS
# -----------------
SYNAPSE_LEARNING_RATE = 0.01        # base learning rate
SYNAPSE_A_PLUS = 0.2               # magnitude of weight potentiation
SYNAPSE_A_MINUS = 0.21             # magnitude of weight depression
SYNAPSE_TAU_PLUS = 20.0            # time constant for potentiation
SYNAPSE_TAU_MINUS = 20.0           # time constant for depression
SYNAPSE_MIN_WEIGHT = 0.0           # minimum synapse weight
SYNAPSE_ELIGIBILITY_DECAY = 0.95   # decay factor for eligibility trace
SYNAPSE_CREATION_THRESHOLD = 0.2   # threshold for synaptogenesis
SYNAPSE_ELIMINATION_THRESHOLD = 0.01 # threshold for synaptic pruning
SYNAPSE_NOISE_FACTOR = 0.01        # small random noise for eligibility

# SYNAPSE WEIGHT INITIALIZATION
# ----------------------------
SYNAPSE_INIT_MIN_WEIGHT = 6.0      # min weight for new synapses (mV)
SYNAPSE_INIT_MAX_WEIGHT = 12.0      # max weight for new synapses (mV)
SYNAPSE_MAX_WEIGHT = 20.0          # max weight allowed during learning

# NETWORK ARCHITECTURE
# ------------------
NETWORK_INPUT_SIZE = 8             # number of input neurons (sensor percepts)
NETWORK_HIDDEN_SIZE = 10           # number of hidden neurons
NETWORK_OUTPUT_SIZE = 4            # number of output neurons (actions)
NETWORK_HIDDEN_RECURRENT_PROB = 0.2 # probability of recurrent connections

# STRUCTURAL PLASTICITY
# -------------------
STRUCTURAL_PLASTICITY_CHECK_INTERVAL = 1000  # time steps between plasticity checks
STRUCTURAL_PLASTICITY_MAX_DENSITY = 0.4      # maximum network density
STRUCTURAL_PLASTICITY_MIN_DENSITY = 0.2      # minimum network density
STRUCTURAL_PLASTICITY_ACTIVITY_THRESHOLD = 50 # activity threshold for new connections
STRUCTURAL_PLASTICITY_MAX_NEW_CONNECTIONS = 5 # max new connections per update

# TRAINING PARAMETERS
# -----------------
TRAINING_NUM_EPISODES = 20         # number of training episodes
TRAINING_MAX_STEPS_PER_EPISODE = 50 # maximum steps per episode

PERCEPT_SCALING = 70.0  # Scale factor to bring percepts into proper mV range

class AdaptiveExponentialLIFNeuron(nn.Module):
    """
    Adaptive Exponential Leaky Integrate-and-Fire Neuron with homeostatic plasticity
    Mimics biological neurons with adaptation mechanisms
    """
    def __init__(self, threshold=NEURON_THRESHOLD, rest_potential=NEURON_REST_POTENTIAL, 
                 reset_potential=NEURON_RESET_POTENTIAL, leak_constant=NEURON_LEAK_CONSTANT, 
                 adaptation_constant=NEURON_ADAPTATION_CONSTANT, refractory_period=NEURON_REFRACTORY_PERIOD,
                 excitatory=True, device=None):
        super(AdaptiveExponentialLIFNeuron, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.rest_potential = rest_potential
        self.reset_potential = reset_potential
        self.leak_constant = leak_constant
        self.adaptation_constant = adaptation_constant
        self.refractory_period = refractory_period
        self.excitatory = excitatory  # Determines if neuron is excitatory or inhibitory
        
        # Homeostatic plasticity parameters
        self.target_firing_rate = NEURON_TARGET_FIRING_RATE
        self.homeostatic_time_constant = NEURON_HOMEOSTATIC_TIME_CONSTANT
        self.homeostatic_learning_rate = NEURON_HOMEOSTATIC_LEARNING_RATE
        
        # Initialize state
        self.reset()
        
    def reset(self):
        self.membrane_potential = torch.tensor(self.rest_potential, dtype=torch.float32, device=self.device)
        self.adaptation_current = 0.0
        self.refractory_countdown = 0
        self.firing_history = deque(maxlen=self.homeostatic_time_constant)
        self.spike_times = []
        self.potential_history = []
        self.last_spike_time = -1000  # Large negative value for initial state
        
    def forward(self, input_current, time_step):
        """Process input current at current time step"""
        # Ensure input_current is a tensor
        if not torch.is_tensor(input_current):
            input_current = torch.tensor(input_current, dtype=torch.float32, device=self.device)
        
        # No arbitrary scaling - use input directly
        # Input currents should already be in appropriate range for the neuron model
        
        # Update membrane potential if not in refractory period
        if self.refractory_countdown <= 0:
            # Leak towards rest potential
            self.membrane_potential = self.leak_constant * (self.membrane_potential - self.rest_potential) + self.rest_potential
            
            # Add input current directly without scaling
            if not torch.is_tensor(self.membrane_potential) or self.membrane_potential.dim() == 0:
                self.membrane_potential = torch.full_like(input_current, self.rest_potential, 
                                                        dtype=torch.float32, device=self.device)
            
            self.membrane_potential = self.membrane_potential + input_current
            
            # Apply adaptation current
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
                self.membrane_potential = torch.tensor(self.reset_potential, dtype=torch.float32, device=self.device)
                self.adaptation_current += self.adaptation_constant
                self.refractory_countdown = self.refractory_period
                self.last_spike_time = time_step
                self.spike_times.append(time_step)
        else:
            # Scalar or single-element tensor case
            try:
                if self.membrane_potential.item() >= self.threshold:
                    spike = 1.0
                    self.membrane_potential = torch.tensor(self.reset_potential, dtype=torch.float32, device=self.device)
                    self.adaptation_current += self.adaptation_constant
                    self.refractory_countdown = self.refractory_period
                    self.last_spike_time = time_step
                    self.spike_times.append(time_step)
            except:
                # Fallback for any other case
                if torch.any(self.membrane_potential >= self.threshold):
                    spike = 1.0
                    self.membrane_potential = torch.tensor(self.reset_potential, dtype=torch.float32, device=self.device)
                    self.adaptation_current += self.adaptation_constant
                    self.refractory_countdown = self.refractory_period
                    self.last_spike_time = time_step
                    self.spike_times.append(time_step)
        
        # Decay adaptation current
        self.adaptation_current *= NEURON_ADAPTATION_DECAY
        
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
    def __init__(self, initial_weight=0.0, learning_rate=SYNAPSE_LEARNING_RATE, 
                 a_plus=SYNAPSE_A_PLUS, a_minus=SYNAPSE_A_MINUS, 
                 tau_plus=SYNAPSE_TAU_PLUS, tau_minus=SYNAPSE_TAU_MINUS,
                 max_weight=SYNAPSE_MAX_WEIGHT, min_weight=SYNAPSE_MIN_WEIGHT, 
                 structural_plasticity=True, synapse_id=None, device=None):
        super(RewardModulatedSTDPSynapse, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Main weight parameter
        self.weight = nn.Parameter(torch.tensor(initial_weight, device=self.device))
        self.learning_rate = learning_rate
        # Increase base learning rate for faster learning
        self.base_learning_rate = learning_rate
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
        self.pre_trace = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.post_trace = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        # Eligibility trace for reward-modulated learning
        self.eligibility_trace = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.eligibility_decay = SYNAPSE_ELIGIBILITY_DECAY
        
        # Debug information
        self.last_weight_change = 0.0
        self.last_eligibility_value = 0.0
        
        # Structural plasticity parameters
        self.structural_plasticity = structural_plasticity
        self.creation_threshold = SYNAPSE_CREATION_THRESHOLD
        self.elimination_threshold = SYNAPSE_ELIMINATION_THRESHOLD
        self.last_usage = 0  # Time step of last significant use
        
        # History for analysis
        self.weight_history = [initial_weight]
        self.eligibility_history = [0.0]
        
    def forward(self, pre_spike, time_step):
        """Propagate spike through the synapse with appropriate scaling"""
        if not self.enabled:
            return 0.0
            
        # Ensure types
        if not isinstance(pre_spike, torch.Tensor):
            pre_spike = torch.tensor(pre_spike, dtype=torch.float32, device=self.device)
        if not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight, dtype=torch.float32, device=self.device)
            
        # Update usage timestamp
        if (pre_spike > 0).any() and (self.weight > 0.1).all():
            self.last_usage = time_step
            
        # Return properly weighted current (no additional scaling needed)
        return pre_spike * self.weight
        
    def update_traces(self, pre_spike, post_spike, time_step):
        """Update STDP traces based on spike timing"""
        if not self.enabled:
            return
            
        # Ensure pre_spike and post_spike are tensors on the correct device
        if not isinstance(pre_spike, torch.Tensor):
            pre_spike = torch.tensor(pre_spike, dtype=torch.float32, device=self.device)
        if not isinstance(post_spike, torch.Tensor):
            post_spike = torch.tensor(post_spike, dtype=torch.float32, device=self.device)
            
        # Update exponential traces with device-aware tensors
        self.pre_trace = self.pre_trace * torch.exp(torch.tensor(-1.0/self.tau_plus, dtype=torch.float32, device=self.device)) + pre_spike
        self.post_trace = self.post_trace * torch.exp(torch.tensor(-1.0/self.tau_minus, dtype=torch.float32, device=self.device)) + post_spike
        
        # Calculate STDP update based on relative timing
        stdp_update = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        # Check if post-synaptic neuron fired (handle both scalar and tensor cases)
        if isinstance(post_spike, torch.Tensor) and post_spike.numel() > 1:
            if torch.any(post_spike > 0):
                stdp_update += self.a_plus * self.pre_trace
        elif post_spike > 0:  # Scalar or single-element tensor case
            stdp_update += self.a_plus * self.pre_trace
        
        # Check if pre-synaptic neuron fired (handle both scalar and tensor cases)
        if isinstance(pre_spike, torch.Tensor) and pre_spike.numel() > 1:
            if torch.any(pre_spike > 0):
                stdp_update -= self.a_minus * self.post_trace
        elif pre_spike > 0:  # Scalar or single-element tensor case
            stdp_update -= self.a_minus * self.post_trace
        
        # Add noise on the correct device
        noise = torch.randn(1, device=self.device).item() * SYNAPSE_NOISE_FACTOR  # Small random noise
        
        # Store source layer and index for wall detection (add these as instance variables)
        if hasattr(self, 'source_layer') and self.source_layer == 'input' and hasattr(self, 'source_idx') and self.source_idx < 8:
            # Check if this synapse is from a wall-detecting input neuron
            from_wall = False
            if hasattr(self, 'last_percept_value') and self.last_percept_value < 0:
                from_wall = True
                # Enhanced eligibility trace for wall learning
                self.eligibility_trace = self.eligibility_trace * self.eligibility_decay + stdp_update * 1.5 + noise
            else:
                # Normal eligibility trace update
                self.eligibility_trace = self.eligibility_trace * self.eligibility_decay + stdp_update + noise
        else:
            # Normal eligibility trace update for non-input synapses
            self.eligibility_trace = self.eligibility_trace * self.eligibility_decay + stdp_update + noise
        
        # Store for debugging
        self.last_eligibility_value = float(self.eligibility_trace) if torch.is_tensor(self.eligibility_trace) else self.eligibility_trace
        
        # Record history
        self.eligibility_history.append(self.last_eligibility_value)
    
    def apply_reward(self, reward_signal):
        """Apply reward-modulated weight change using eligibility trace"""
        if not self.enabled:
            return
            
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
            # if abs(weight_change) > 0.001:
            #     print(f"Weight changed by {weight_change:.4f} to {self.weight.item():.4f} (elig:{eligibility_value:.4f}, reward:{reward_signal:.2f})")
    
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
    
class MazeBrain(nn.Module):
    """
    Neural network brain for a maze-solving creature.
    """
    def __init__(self, hidden_size=NETWORK_HIDDEN_SIZE, device=None):
        super(MazeBrain, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = NETWORK_INPUT_SIZE
        self.hidden_size = hidden_size
        self.output_size = NETWORK_OUTPUT_SIZE

        # Define comprehensive neuron parameters based on biologically plausible values
        self.neuron_kwargs = {
            'threshold': NEURON_THRESHOLD,
            'rest_potential': NEURON_REST_POTENTIAL,
            'reset_potential': NEURON_RESET_POTENTIAL,
            'leak_constant': NEURON_LEAK_CONSTANT,
            'adaptation_constant': NEURON_ADAPTATION_CONSTANT,
            'refractory_period': NEURON_REFRACTORY_PERIOD,
            'excitatory': True,
            'device': self.device
        }

        # Input neurons (all excitatory)
        self.input_neurons = nn.ModuleList([
            AdaptiveExponentialLIFNeuron(**self.neuron_kwargs) for _ in range(self.input_size)
        ])
        
        # Hidden neurons (80% excitatory, 20% inhibitory)
        hidden_kwargs = self.neuron_kwargs.copy()
        self.hidden_neurons = nn.ModuleList([
            AdaptiveExponentialLIFNeuron(
                **{**hidden_kwargs, 'excitatory': (i < int(0.8 * hidden_size))}) 
            for i in range(hidden_size)
        ])
        
        # Output neurons (all excitatory)
        # Reduce threshold for output neurons to encourage more firing
        self.output_neurons = nn.ModuleList([
            AdaptiveExponentialLIFNeuron(
                **{**self.neuron_kwargs, 'threshold': NEURON_THRESHOLD * 0.9}) 
            for _ in range(self.output_size)
        ])
        
        # Create synapses
        self.synapses = {}
        self.next_synapse_id = 0
        self._initialize_connectivity()

        self.time_step = 0
        self.reward_buffer = deque(maxlen=100)
        self.activity_level = 0.0
        self.last_action = None  # Store the last taken action

        # Move the entire brain to the device
        self.to(self.device)

    def _initialize_connectivity(self):
        # Input to hidden (dense)
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                self._create_synapse('input', i, 'hidden', h)
        # Hidden to hidden (sparse recurrent)
        for h1 in range(self.hidden_size):
            for h2 in range(self.hidden_size):
                if h1 != h2 and random.random() < NETWORK_HIDDEN_RECURRENT_PROB:
                    self._create_synapse('hidden', h1, 'hidden', h2)
        # Hidden to output (dense)
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                self._create_synapse('hidden', h, 'output', o)


    def _create_synapse(self, source_layer, source_idx, target_layer, target_idx, initial_weight=None):
        # Get the source neuron to check if it's excitatory or inhibitory
        source_neuron = self._get_neuron(source_layer, source_idx)
        
        if initial_weight is None:
            # Increase initial weights to drive more activity
            base_weight = random.uniform(SYNAPSE_INIT_MIN_WEIGHT * 1.2, SYNAPSE_INIT_MAX_WEIGHT * 1.2)
            
            # If source neuron is inhibitory, make weight negative with appropriate strength
            if hasattr(source_neuron, 'excitatory') and not source_neuron.excitatory:
                initial_weight = -base_weight * 1.5  # Make inhibitory neurons slightly stronger
            else:
                initial_weight = base_weight
        
        synapse = RewardModulatedSTDPSynapse(
            initial_weight=initial_weight,
            max_weight=SYNAPSE_MAX_WEIGHT if initial_weight >= 0 else -SYNAPSE_MIN_WEIGHT,
            min_weight=SYNAPSE_MIN_WEIGHT if initial_weight >= 0 else -SYNAPSE_MAX_WEIGHT,
            synapse_id=self.next_synapse_id,
            device=self.device
        )
        synapse.source_layer = source_layer
        synapse.source_idx = source_idx
        synapse.target_layer = target_layer
        synapse.target_idx = target_idx
        self.next_synapse_id += 1
        key = (source_layer, source_idx, target_layer, target_idx)
        self.synapses[key] = synapse
        return synapse

    def _get_neuron(self, layer, idx):
        if layer == 'input':
            return self.input_neurons[idx]
        elif layer == 'hidden':
            return self.hidden_neurons[idx]
        elif layer == 'output':
            return self.output_neurons[idx]
        raise ValueError(f"Unknown layer: {layer}")

    def forward(self, percepts, reward_signal=0.0):
        """
        Process percepts and return action (direction).
        """
        self.time_step += 1

        # 1. Process Input Layer (Percepts)
        input_spikes = []
        for i, percept in enumerate(percepts):
            # Walls (-1) become inhibitory but not extreme, exits (1) become excitatory
            if percept < 0:  # Wall
                scaled_percept = -25.0  # Reduced inhibition (was -40.0)
            elif percept > 0:  # Exit
                scaled_percept = 30.0  # Fixed value instead of scaling
            else:  # Free space (0)
                scaled_percept = 0
            spike, _ = self.input_neurons[i](scaled_percept, self.time_step)
            input_spikes.append(spike)
            # Store percept value in all synapses coming from this input neuron
            for h in range(self.hidden_size):
                key = ('input', i, 'hidden', h)
                if key in self.synapses:
                    self.synapses[key].last_percept_value = percept

        # 2. Process Hidden Layer
        hidden_inputs = [0.0] * self.hidden_size
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                key = ('input', i, 'hidden', h)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    current = synapse(input_spikes[i], self.time_step)
                    hidden_inputs[h] += current
                    # Register synapse firing for visualization if current exists
                    if current > 0 and hasattr(self, 'env') and self.env:
                        self.env.register_synapse_firing('input', i, 'hidden', h)

        hidden_spikes = []
        for h in range(self.hidden_size):
            spike, _ = self.hidden_neurons[h](hidden_inputs[h], self.time_step)
            hidden_spikes.append(spike)

        # Recurrent connections within hidden layer
        for h1 in range(self.hidden_size):
            if hidden_spikes[h1] > 0:
                for h2 in range(self.hidden_size):
                    key = ('hidden', h1, 'hidden', h2)
                    if key in self.synapses:
                         synapse = self.synapses[key]
                         current = synapse(hidden_spikes[h1], self.time_step)
                         self.hidden_neurons[h2].membrane_potential += current
                         # Register synapse firing for visualization
                         if current > 0 and hasattr(self, 'env') and self.env:
                             self.env.register_synapse_firing('hidden', h1, 'hidden', h2)

        # 3. Process Output Layer (Action)
        output_inputs = [0.0] * self.output_size
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                key = ('hidden', h, 'output', o)
                if key in self.synapses:
                    synapse = self.synapses[key]
                    current = synapse(hidden_spikes[h], self.time_step)
                    output_inputs[o] += current
                    # Register synapse firing for visualization
                    if current > 0 and hasattr(self, 'env') and self.env:
                        self.env.register_synapse_firing('hidden', h, 'output', o)

        output_spikes = []
        for o in range(self.output_size):
            spike, _ = self.output_neurons[o](output_inputs[o], self.time_step)
            output_spikes.append(spike)

        # 4. Choose Action (Highest spiking output neuron)
        if any(output_spikes):  # Check if any output neuron fired
            max_activation = max(output_spikes)
            max_indices = [i for i, spike in enumerate(output_spikes) if spike == max_activation]
            action_index = random.choice(max_indices)  # Randomly select among maximum activations
            self.last_action = action_index  # Up, Down, Left, Right
        else:
            action_index = random.randint(0, 3)  # Take random action instead of returning None
            self.last_action = action_index

        # 5. Learning and Update
        self._update_synapse_traces(input_spikes, hidden_spikes, output_spikes)
        if reward_signal != 0:
             self.reward_buffer.append(reward_signal)
             self._apply_reward(reward_signal)

        #Store overall network activity level
        self.activity_level = 0.6 * self.activity_level + 0.4 * (
            sum(input_spikes) / len(input_spikes) +
            sum(hidden_spikes) / len(hidden_spikes) +
            sum(output_spikes) / len(output_spikes)
        ) / 3.0

        # Perform structural plasticity (synaptogenesis & pruning)
        if self.time_step % STRUCTURAL_PLASTICITY_CHECK_INTERVAL == 0:  # Check every 1000 time steps
            self._structural_plasticity()
        return action_index

    def connect_to_environment(self, env):
        """Connect this brain to an environment for visualization"""
        self.env = env
        env.set_brain(self)

    def _update_synapse_traces(self, input_spikes, hidden_spikes, output_spikes):
        """Update all synapse traces."""
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                key = ('input', i, 'hidden', h)
                if key in self.synapses:
                    self.synapses[key].update_traces(input_spikes[i], hidden_spikes[h], self.time_step)
        for h1 in range(self.hidden_size):
            for h2 in range(self.hidden_size):
                key = ('hidden', h1, 'hidden', h2)
                if key in self.synapses:
                    self.synapses[key].update_traces(hidden_spikes[h1], hidden_spikes[h2], self.time_step)
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                key = ('hidden', h, 'output', o)
                if key in self.synapses:
                    self.synapses[key].update_traces(hidden_spikes[h], output_spikes[o], self.time_step)

    def _apply_reward(self, reward_signal):
        # For negative rewards (wall hits), strengthen learning
        reward_multiplier = 2.0 if reward_signal < 0 else 1.0
        
        for key, synapse in self.synapses.items():
            source_layer, source_idx, target_layer, target_idx = key
            
            # Apply stronger learning for wall avoidance
            if reward_signal < 0:
                # Stronger penalty for the action that led to wall collision
                if target_layer == 'output' and target_idx == self.last_action:
                    synapse.apply_reward(reward_signal * 2.5 * reward_multiplier)
                else:
                    synapse.apply_reward(reward_signal * 0.8 * reward_multiplier)
            else:
                # Regular reward processing for positive/neutral rewards
                if target_layer == 'output' and target_idx == self.last_action:
                    synapse.apply_reward(reward_signal * 1.5)
                else:
                    synapse.apply_reward(reward_signal * 0.8)

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
        if current_density < STRUCTURAL_PLASTICITY_MAX_DENSITY:  # Keep overall density in check
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
                if activity > STRUCTURAL_PLASTICITY_ACTIVITY_THRESHOLD and neuron_connections[neuron] < 5:
                    candidates.append(neuron)
            
            # Create new connections to these neurons
            for target in candidates[:STRUCTURAL_PLASTICITY_MAX_NEW_CONNECTIONS]:  # Limit to 5 new connections per update
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

    def reset(self):
        """Resets the brain to an initial state."""
        for neuron in self.input_neurons:
            neuron.reset()
        for neuron in self.hidden_neurons:
            neuron.reset()
        for neuron in self.output_neurons:
            neuron.reset()

        self.time_step = 0
        self.reward_buffer.clear()
        self.activity_level = 0.0
        self.last_action = None

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    import pygame
    from maze_environment import MazeEnvironment
    
    # Create brain and environment
    brain = MazeBrain()
    env = MazeEnvironment()
    
    # Connect brain to environment for visualization
    brain.connect_to_environment(env)
    
    # Training loop
    num_episodes = TRAINING_NUM_EPISODES
    max_steps_per_episode = TRAINING_MAX_STEPS_PER_EPISODE
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps_per_episode):
                # Get action from brain
                action = brain(state)
                
                # If no action determined, choose randomly
                if action is None:
                    action = random.randint(0, 3)
                
                # Take action in environment
                next_state, reward, done = env.step(action)
                
                # Learn from the experience
                brain(next_state, reward_signal=reward)
                
                # Update state and accumulate reward
                state = next_state
                episode_reward += reward
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                
                # Break if episode is done
                if done:
                    print(f"Episode {episode+1} completed in {step+1} steps with reward {episode_reward}")
                    break
            
            # If episode didn't reach goal
            if not done:
                print(f"Episode {episode+1} failed to reach goal in {max_steps_per_episode} steps")
        
        # Keep visualization open at the end
        print("Training complete. Press any key to exit.")
        waiting = True
        while waiting:
            env.visualize()  # Keep updating visualization
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
    
    finally:
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()