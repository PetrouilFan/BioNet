# BioNet: A Continuously Adapting Spiking Neural Network System

## Project Overview
BioNet is an innovative implementation of a brain-inspired artificial intelligence system based on Spiking Neural Networks (SNNs) with continuous adaptation capabilities. Unlike traditional neural networks that rely on fixed architectures and static training phases, BioNet aims to create a self-organizing, continuously adapting neural system that can learn and evolve throughout its operational lifetime.

### Core Problems Solved:

- Overcomes the limitations of traditional static neural networks through continuous adaptation
- Enables dynamic restructuring of neural pathways based on experience
- Facilitates transfer learning across different domains and tasks
- Implements biologically plausible learning mechanisms for improved generalization

### Key Technologies:

- PyTorch-based SNN implementation
- Spike-Timing-Dependent Plasticity (STDP)
- Reward-modulated learning
- Dynamic structural plasticity (synaptogenesis and pruning)
- Modular neural architecture with cross-module connectivity

## Architecture Overview
BioNet implements a modular, event-driven architecture inspired by the structural and functional organization of biological neural systems.

### Core Components
```
┌───────────────────────────────────────────────────────────────┐
│                ContinuouslyAdaptingNeuralSystem               │
├───────────┬───────────────┬───────────────┬───────────────────┤
│Perception │  Association  │     Motor     │    Additional     │
│  Module   │    Module     │    Module     │     Modules       │
├───────────┴───────────────┴───────────────┴───────────────────┤
│                      Cross-module Connections                 │
├─────────────────────────────┬─────────────────────────────────┤
│        Neural Modules       │          Environment            │
├─────────────────┬───────────┼─────────────────┬───────────────┤
│  Input Neurons  │  Synapses │  Sensory Queue  │ Action Queue  │
├─────────────────┼───────────┼─────────────────┼───────────────┤
│ Hidden Neurons  │   STDP    │  Reward Signal  │  Visualization│
├─────────────────┼───────────┼─────────────────┼───────────────┤
│ Output Neurons  │Plasticity │   Evaluation    │  Performance  │
└─────────────────┴───────────┴─────────────────┴───────────────┘
```

### Neurons:

* **AdaptiveExponentialLIFNeuron**: Implements the Adaptive Exponential Leaky Integrate-and-Fire model with homeostatic plasticity
* Features refractory periods, adaptation mechanisms, and threshold adjustments

### Synapses:

* **RewardModulatedSTDPSynapse**: Implements spike-timing-dependent plasticity with reward modulation
* Maintains eligibility traces for reinforcement learning
* Supports dynamic weight adjustment and structural plasticity

### Neural Modules:

* Self-contained neural networks with input, hidden, and output layers
* Specialized for specific functions (perception, association, motor control)
* Support sparse, recurrent connectivity patterns

### Neural System:

* **ContinuouslyAdaptingNeuralSystem**: Coordinates multiple neural modules
* Manages cross-module connections for information transfer
* Implements continuous adaptation and structural modifications

### Environments:

* Testing environments for evaluating neural system performance
* **MazeEnvironment**: For spatial navigation tasks
* **SimpleEnvironment**: For basic reinforcement learning scenarios

## Data Flow

### Sensory Input Flow:
```
External Stimuli → Sensory Queue → Perception Module(s) → 
Association Module(s) → Motor Module(s) → Action Queue → Environment
```

### Reward Signal Flow:
```
Environment → Reward Signal → Neural System → 
Synapse Eligibility Traces → Weight Updates
```

### Adaptation Flow:
```
Performance Evaluation → Structural Plasticity → 
Synaptogenesis/Pruning → Connection Density Adjustment
```

## Design Patterns

* **Observer Pattern**: Modules observe and respond to spikes and rewards
* **Factory Pattern**: Dynamic creation of neurons and synapses
* **Dependency Injection**: Environments and visualizers are injected into the neural system
* **Command Pattern**: Actions are queued for execution in the environment

## Technical Implementation

### Neural Components

#### Spiking Neurons
The system uses the Adaptive Exponential Leaky Integrate-and-Fire (AdEx-LIF) model, which combines biological plausibility with computational efficiency:

```python
class AdaptiveExponentialLIFNeuron(nn.Module):
    def forward(self, input_current, time_step):
        # Update membrane potential based on input current and leakage
        if not self.refractory_countdown:
            self.membrane_potential = (self.leak_constant * self.membrane_potential + 
                                       (1 - self.leak_constant) * self.rest_potential + 
                                       input_current - self.adaptation_current)
            
            # Generate spike if threshold is reached
            if self.membrane_potential >= self.threshold:
                spike = 1.0
                self.membrane_potential = self.reset_potential
                self.adaptation_current += self.adaptation_constant
                self.refractory_countdown = self.refractory_period
                self.last_spike_time = time_step
                self.spike_times.append(time_step)
        
        # Implement homeostatic plasticity by adjusting threshold
        self._adjust_threshold()
        
        return spike, self.membrane_potential
```

#### Learning Mechanisms
The system implements a biologically-inspired Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP):

```python
class RewardModulatedSTDPSynapse(nn.Module):
    def update_traces(self, pre_spike, post_spike, time_step):
        # Update pre-synaptic trace
        self.pre_trace *= torch.exp(-1.0 / self.tau_plus)
        if pre_spike > 0:
            self.pre_trace += 1.0
            
        # Update post-synaptic trace
        self.post_trace *= torch.exp(-1.0 / self.tau_minus)
        if post_spike > 0:
            self.post_trace += 1.0
            
        # Update eligibility trace based on STDP rules
        if pre_spike > 0:  # Pre-synaptic neuron fired
            self.eligibility_trace += self.a_minus * self.post_trace
            
        if post_spike > 0:  # Post-synaptic neuron fired
            self.eligibility_trace += self.a_plus * self.pre_trace
            
        # Decay eligibility trace
        self.eligibility_trace *= self.eligibility_decay
```

#### Structural Plasticity
The system implements dynamic structural plasticity through synaptogenesis (creation of new synapses) and pruning (removal of unused synapses):

```python
def _structural_plasticity(self):
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
    
    # Remove pruned synapses
    for key in keys_to_remove:
        del self.synapses[key]
        
    # Synaptogenesis - create new synapses where needed
    if current_density < STRUCTURAL_PLASTICITY_MAX_DENSITY:
        # Find underconnected neurons with high activity
        neuron_activity = {}
        neuron_connections = {}
        
        # Analyze hidden neurons
        for h in range(self.hidden_size):
            activity = sum(1 for spike in self.hidden_neurons[h].firing_history if spike > 0)
            neuron_activity[('hidden', h)] = activity
            neuron_connections[('hidden', h)] = 0
        
        # Create new connections to active neurons
        candidates = []
        for neuron, activity in neuron_activity.items():
            if activity > STRUCTURAL_PLASTICITY_ACTIVITY_THRESHOLD and neuron_connections[neuron] < 5:
                candidates.append(neuron)
        
        for target in candidates[:STRUCTURAL_PLASTICITY_MAX_NEW_CONNECTIONS]:
            # Create new connections to this neuron
            # ...
```

#### Modular Architecture
The system is designed with modularity as a core principle, allowing different neural modules to specialize in specific functions:

```python
class NeuralModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, module_name="generic"):
        super(NeuralModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.module_name = module_name
        
        # Create neurons for each layer (80% excitatory, 20% inhibitory)
        self.input_neurons = nn.ModuleList([...])
        self.hidden_neurons = nn.ModuleList([...])
        self.output_neurons = nn.ModuleList([...])
```

Different specialized modules:

```python
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
```

#### Visualization and Monitoring
The system includes comprehensive visualization tools for monitoring neural activity, network structure, and learning performance:

```python
class Visualizer:
    """Visualizes neural activity and network structure"""
    def __init__(self, neural_system):
        self.neural_system = neural_system
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Initialize visualization components
        self.module_activity = {name: [] for name in self.neural_system.modules_dict}
        self.connection_strength = []
        self.rewards_history = []
        self.time_points = []
```

## Development Guide

### Codebase Organization

* **continues.py**: Core neural system components
  * AdaptiveExponentialLIFNeuron: Neuron implementation
  * RewardModulatedSTDPSynapse: Synapse implementation with learning
  * NeuralModule: Base class for modular neural networks
  * ContinuouslyAdaptingNeuralSystem: System coordinator

* **main.py**: System initialization and demonstration
  * PerceptionModule, AssociationModule, MotorModule: Specialized modules
  * Visualizer: Neural activity visualization
  * SimpleEnvironment: Basic testing environment

* **demo_maze.py**: Maze navigation demonstration
  * MazeBrain: Specialized neural network for maze solving
  * Parameters and constants for the maze environment

* **maze_environment.py**: Maze environment implementation
  * MazeEnvironment: Environment with walls, agent, and goals
  * Visualization of the maze and neural network

* **debug_synapses.py**: Debugging utilities for synapse behavior

### Key Concepts and Abstractions

#### Spiking Neurons vs Traditional Neurons
Unlike traditional artificial neurons that output continuous activation values, spiking neurons emit discrete spikes based on membrane potential:

| Traditional Neurons | Spiking Neurons |
|---------------------|----------------|
| Continuous activation | Binary spikes |
| Synchronous updates | Event-driven updates |
| Static parameters | Adaptive parameters |
| Batch processing | Online processing |

#### Plasticity Mechanisms
The system implements multiple forms of neural plasticity:

1. **Hebbian Plasticity**: Neurons that fire together, wire together
2. **Homeostatic Plasticity**: Neurons adjust their excitability to maintain stable activity
3. **Structural Plasticity**: Formation and pruning of synapses
4. **Reward-Modulated Plasticity**: Learning guided by reward signals

#### Continuous Adaptation Loop
```
┌──────────┐      ┌──────────┐     ┌────────────┐
│Perception│────▶│Processing │───▶│   Action   │
└────┬─────┘      └───┬──────┘     └─────┬──────┘
     │                │                  │
     │                ▼                  │
     │          ┌──────────┐             │
     │          │ Learning │             │
     │          └────┬─────┘             │
     │               │                   │
     ▼               ▼                   ▼
┌───────────────────────────────────────────┐
│                Environment                │
└───────────────┬───────────────────────────┘
                │
                ▼
          ┌──────────┐
          │  Reward  │
          └──────────┘
```

### Extension Points

#### Add New Modules
Create specialized neural modules for different cognitive functions:

```python
class MemoryModule(NeuralModule):
    def __init__(self, input_size=10, hidden_size=30, output_size=10):
        super(MemoryModule, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            module_name="memory"
        )
        self.has_recurrence = True
```

#### Create New Environments
Implement environments for specific testing scenarios:

```python
class RoboticsEnvironment(Environment):
    def __init__(self, neural_system):
        super(RoboticsEnvironment, self).__init__()
        # Initialize robotics simulation
```

#### Implement New Learning Rules
Add alternative synaptic plasticity mechanisms:

```python
class BCMSynapse(nn.Module):
    # Implement Bienenstock-Cooper-Munro learning rule
```

### Testing Approach

* **Component Testing**: Individual neurons and synapses can be tested with unit tests
* **Integration Testing**: Test interaction between modules
* **Environment-Based Testing**: Run the system in different environments
* **Visual Debugging**: Use the visualization tools to monitor and debug

## Current Project Status

This project is a work in progress with several features at different stages of implementation:

### Implemented Features

- ✅ Spiking neural network architecture with AdEx-LIF neurons
- ✅ Reward-modulated STDP learning
- ✅ Basic structural plasticity (synaptogenesis and pruning)
- ✅ Modular neural architecture
- ✅ Simple testing environments (SimpleEnvironment, MazeEnvironment)
- ✅ Real-time visualization of neural activity

### In Development

- 🔄 Cross-domain transfer learning
- 🔄 Meta-learning capabilities
- 🔄 More sophisticated structural plasticity rules
- 🔄 Enhanced reward signal distribution
- 🔄 Advanced homeostatic mechanisms

### Future Ideas

- 📅 Hardware acceleration for neuromorphic computing
- 📅 Integration with robotic systems
- 📅 Multi-agent interaction capabilities
- 📅 Deployment on specialized neural processors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.